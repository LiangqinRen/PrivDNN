#include "common.h"
#include "configor/json.hpp"
#include "timer.h"
#include "utils.h"

#include <array>
#include <fstream>
#include <map>
#include <typeinfo>
#include <variant>

extern "C" {

size_t get_index(const array<size_t, 4> &shape, const array<size_t, 4> &indexes) {
    size_t index = 0, d_size = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        index += indexes[i] * d_size;
        d_size *= shape[i];
    }

    return index;
}

void degrade_cipher_levels(SEALPACK &seal, Ciphertext &cipher, size_t level) {
    double one = 1.0;
    Plaintext one_plain;
    seal.encoder_.encode(one, SCALE, one_plain);

    for (size_t i = 0; i < level; ++i) {
        parms_id_type last_parms_id = cipher.parms_id();
        seal.evaluator_.mod_switch_to_inplace(one_plain, last_parms_id);
        seal.evaluator_.multiply_plain_inplace(cipher, one_plain);
        seal.evaluator_.rescale_to_next_inplace(cipher);
    }
}

vector<variant<double, Ciphertext>> read_conv_weight(
    string dataset,
    SEALPACK &seal,
    int round,
    json encrypted_neurons) {
    MNIST_Shape shapes;
    auto weight_shape = shapes.conv_weight[round];
    size_t weight_size = 1;
    for (size_t i = 0; i < weight_shape.size(); ++i) {
        weight_size *= weight_shape[i];
    }
    vector<variant<double, Ciphertext>> weight(weight_size);

    hash<json> encrypted_list_hash;
    auto files_prefix = to_string(encrypted_list_hash(encrypted_neurons));
    const string weight_path = move(
        DATA_PATH + string(dataset) + string("/") + files_prefix + string("_conv") +
        to_string(round) + string("_weight"));
    ifstream trained_data_input_stream;
    trained_data_input_stream.open(weight_path, ios::in | ios::binary);

    size_t block_size = 1;
    for (size_t i = 1; i < weight_shape.size(); ++i) {
        block_size *= weight_shape[i];
    }

    size_t index = 0;
    for (size_t i = 0; i < weight_shape[0]; ++i) {
        if (is_neuron_encrypted(encrypted_neurons, round, i)) {
            for (size_t j = 0; j < block_size; ++j) {
                seal.cipher_.load(seal.context_, trained_data_input_stream);
                weight[index++] = seal.cipher_;
            }
        } else {
            for (size_t j = 0; j < block_size; ++j) {
                double weight_value = 0;
                trained_data_input_stream.read(
                    reinterpret_cast<char *>(&weight_value), sizeof(weight_value));
                weight[index++] = weight_value;
            }
        }
    }

    trained_data_input_stream.close();
    return weight;
}

vector<variant<double, Ciphertext>> read_conv_bias(
    string dataset,
    SEALPACK &seal,
    int round,
    json encrypted_neurons) {
    MNIST_Shape shapes;
    hash<json> encrypted_list_hash;
    auto files_prefix = to_string(encrypted_list_hash(encrypted_neurons));
    const string bias_path = move(
        DATA_PATH + string(dataset) + string("/") + files_prefix + string("_conv") +
        to_string(round) + string("_bias"));

    ifstream MNIST_bias_instream;
    MNIST_bias_instream.open(bias_path, ios::in | ios::binary);
    vector<variant<double, Ciphertext>> bias(shapes.conv_bias[round]);
    for (size_t i = 0; i < bias.size(); ++i) {
        if (is_neuron_encrypted(encrypted_neurons, round, i)) {
            seal.cipher_.load(seal.context_, MNIST_bias_instream);
            degrade_cipher_levels(seal, seal.cipher_, 1 + (round - 1) * 3);
            bias[i] = seal.cipher_;
        } else {
            double bias_value;
            MNIST_bias_instream.read(reinterpret_cast<char *>(&bias_value), sizeof(bias_value));
            bias[i] = bias_value;
        }
    }
    MNIST_bias_instream.close();

    return bias;
}

void thread_conv_worker(
    SEALPACK &seal,
    const vector<array<size_t, 4>> &output_indexes,
    const vector<variant<vector<double>, Ciphertext>> &input,
    const vector<variant<double, Ciphertext>> &weight,
    Ciphertext bias,
    vector<variant<vector<double>, Ciphertext>> &result,
    MNIST_Shape shapes,
    size_t round,
    int begining,
    int ending) {
    auto encrypted_neurons = get_encrypted_neurons_list(string("MNIST"));

    for (int i = begining; i < ending; ++i) {
        auto sum = bias;
        if (round == 1) {
            for (size_t a = 0; a < shapes.kernal_sizes[1]; ++a) {
                for (size_t b = 0; b < shapes.kernal_sizes[1]; ++b) {
                    auto input_values = get<vector<double>>(input[get_index(
                        shapes.conv_input[1],
                        {0, 0, output_indexes[i][2] + a, output_indexes[i][3] + b})]);
                    Plaintext input_plain;
                    seal.encoder_.encode(input_values, SCALE, input_plain);
                    auto weight_cipher = get<Ciphertext>(
                        weight[get_index(shapes.conv_weight[1], {output_indexes[i][1], 0, a, b})]);

                    Ciphertext multiplied_cipher;
                    seal.evaluator_.multiply_plain(weight_cipher, input_plain, multiplied_cipher);
                    seal.evaluator_.rescale_to_next_inplace(multiplied_cipher);
                    seal.evaluator_.add_inplace(sum, multiplied_cipher);
                }
            }

            result[get_index(
                shapes.conv_output[1],
                {0, output_indexes[i][1], output_indexes[i][2], output_indexes[i][3]})] = move(sum);
        } else {
            for (size_t channel = 0; channel < shapes.conv_output[1][1]; ++channel) {
                if (!is_neuron_encrypted(encrypted_neurons, round - 1, channel)) {
                    for (size_t a = 0; a < shapes.kernal_sizes[round]; ++a) {
                        for (size_t b = 0; b < shapes.kernal_sizes[round]; ++b) {
                            auto input_values = get<vector<double>>(input[get_index(
                                shapes.conv_input[round],
                                {0, channel, output_indexes[i][2] + a, output_indexes[i][3] + b})]);
                            Plaintext input_plain;
                            seal.encoder_.encode(input_values, SCALE, input_plain);

                            auto weight_cipher = get<Ciphertext>(weight[get_index(
                                shapes.conv_weight[round], {output_indexes[i][1], channel, a, b})]);

                            seal.evaluator_.multiply_plain_inplace(weight_cipher, input_plain);
                            seal.evaluator_.rescale_to_next_inplace(weight_cipher);

                            degrade_cipher_levels(seal, weight_cipher, 3);

                            sum.scale() = weight_cipher.scale() = SCALE;
                            seal.evaluator_.add_inplace(sum, weight_cipher);
                        }
                    }
                } else {
                    for (size_t a = 0; a < shapes.kernal_sizes[round]; ++a) {
                        for (size_t b = 0; b < shapes.kernal_sizes[round]; ++b) {
                            auto input_cipher = get<Ciphertext>(input[get_index(
                                shapes.conv_input[round],
                                {0, channel, output_indexes[i][2] + a, output_indexes[i][3] + b})]);

                            auto weight_cipher = get<Ciphertext>(weight[get_index(
                                shapes.conv_weight[round], {output_indexes[i][1], channel, a, b})]);

                            degrade_cipher_levels(seal, weight_cipher, 3);

                            seal.evaluator_.multiply_inplace(weight_cipher, input_cipher);
                            seal.evaluator_.relinearize_inplace(weight_cipher, seal.relin_keys_);
                            seal.evaluator_.rescale_to_next_inplace(weight_cipher);

                            sum.scale() = weight_cipher.scale() = SCALE;
                            seal.evaluator_.add_inplace(sum, weight_cipher);
                        }
                    }
                }
            }

            result[get_index(
                shapes.conv_output[round],
                {0, output_indexes[i][1], output_indexes[i][2], output_indexes[i][3]})] = move(sum);
        }
    }
}

void thread_conv(
    SEALPACK &seal,
    const vector<array<size_t, 4>> &output_indexes,
    const vector<variant<vector<double>, Ciphertext>> &input,
    const vector<variant<double, Ciphertext>> &weight,
    Ciphertext bias,
    vector<variant<vector<double>, Ciphertext>> &result,
    MNIST_Shape shapes,
    size_t round) {
    const size_t processor_count = thread::hardware_concurrency();
    vector<size_t> threads_task_count(processor_count, output_indexes.size() / processor_count);
    threads_task_count[0] += output_indexes.size() % processor_count;

    int beginnig = 0;
    vector<thread> threads(processor_count);
    for (size_t i = 0; i < processor_count; ++i) {
        threads[i] = thread(
            thread_conv_worker,
            ref(seal),
            cref(output_indexes),
            cref(input),
            cref(weight),
            bias,
            ref(result),
            shapes,
            round,
            beginnig,
            beginnig + threads_task_count[i]);
        beginnig += threads_task_count[i];
    }

    for (size_t i = 0; i < threads.size(); ++i) {
        threads[i].join();
    }
}

vector<variant<vector<double>, Ciphertext>> conv(
    string dataset,
    SEALPACK &seal,
    MNIST_Shape shapes,
    int round,
    vector<variant<vector<double>, Ciphertext>> &input,
    json encrypted_neurons,
    mode work_mode) {
    if (dataset == string("MNIST")) {
        auto conv_weight = read_conv_weight(dataset, seal, round, encrypted_neurons);
        auto conv_bias = read_conv_bias(dataset, seal, round, encrypted_neurons);

        size_t kernal_size = shapes.kernal_sizes[round];
        array<size_t, 4> output_shape = shapes.conv_output[round];
        const size_t batch_size = output_shape[0];

        size_t output_size = 1;
        for (size_t i = 1; i < output_shape.size(); ++i) {
            output_size *= output_shape[i];
        }
        vector<variant<vector<double>, Ciphertext>> output(output_size);

        if (round == 1) {
            for (size_t i = 0; i < output_shape[1]; ++i) {
                if (is_neuron_encrypted(encrypted_neurons, round, i)) {
                    if (work_mode == remove_) {
                        for (size_t j = 0; j < output_shape[2]; ++j) {
                            for (size_t k = 0; k < output_shape[3]; ++k) {
                                output[get_index(output_shape, {0, i, j, k})] =
                                    vector<double>(batch_size);
                            }
                        }
                        continue;
                    }

                    auto bias = std::get<Ciphertext>(conv_bias[i]);

                    vector<array<size_t, 4>> work_indexes;
                    for (size_t j = 0; j < output_shape[2]; ++j) {
                        for (size_t k = 0; k < output_shape[3]; ++k) {
                            work_indexes.emplace_back(array<size_t, 4>{0, i, j, k});
                        }
                    }

                    thread_conv(
                        seal, work_indexes, input, conv_weight, bias, output, shapes, round);

                } else {
                    for (size_t j = 0; j < output_shape[2]; ++j) {
                        for (size_t k = 0; k < output_shape[3]; ++k) {
                            double bias = get<double>(conv_bias[i]);
                            vector<double> multiplied_result(batch_size, bias);
                            for (size_t a = 0; a < kernal_size; ++a) {
                                for (size_t b = 0; b < kernal_size; ++b) {
                                    auto input_values = get<vector<double>>(input[get_index(
                                        shapes.conv_input[1], {0, 0, j + a, k + b})]);
                                    double weight_value = get<double>(conv_weight[get_index(
                                        shapes.conv_weight[1], {i, 0, a, b})]);
                                    for (size_t l = 0; l < batch_size; ++l) {
                                        multiplied_result[l] += input_values[l] * weight_value;
                                    }
                                }
                            }

                            output[get_index(output_shape, {0, i, j, k})] = move(multiplied_result);
                        }
                    }
                }
            }
        } else {
            for (size_t i = 0; i < output_shape[1]; ++i) {
                if (is_neuron_encrypted(encrypted_neurons, round, i)) {
                    if (work_mode == remove_) {
                        for (size_t j = 0; j < output_shape[2]; ++j) {
                            for (size_t k = 0; k < output_shape[3]; ++k) {
                                output[get_index(output_shape, {0, i, j, k})] =
                                    vector<double>(batch_size);
                            }
                        }
                        continue;
                    }

                    Ciphertext bias = get<Ciphertext>(conv_bias[i]);

                    vector<array<size_t, 4>> work_indexes;
                    for (size_t j = 0; j < output_shape[2]; ++j) {
                        for (size_t k = 0; k < output_shape[3]; ++k) {
                            work_indexes.emplace_back(array<size_t, 4>{0, i, j, k});
                        }
                    }

                    thread_conv(
                        seal, work_indexes, input, conv_weight, bias, output, shapes, round);
                } else {
                    for (size_t j = 0; j < output_shape[2]; ++j) {
                        for (size_t k = 0; k < output_shape[3]; ++k) {
                            double bias = get<double>(conv_bias[i]);
                            vector<double> multiplied_result(batch_size, bias);
                            for (size_t channel = 0; channel < shapes.conv_output[1][1];
                                 ++channel) {
                                if (!is_neuron_encrypted(encrypted_neurons, round - 1, channel)) {
                                    for (size_t a = 0; a < kernal_size; ++a) {
                                        for (size_t b = 0; b < kernal_size; ++b) {
                                            auto input_values = get<vector<double>>(input[get_index(
                                                shapes.conv_input[round],
                                                {0, channel, j + a, k + b})]);
                                            double weight_value = get<double>(conv_weight[get_index(
                                                shapes.conv_weight[round], {i, channel, a, b})]);
                                            for (size_t l = 0; l < batch_size; ++l) {
                                                multiplied_result[l] +=
                                                    input_values[l] * weight_value;
                                            }
                                        }
                                    }
                                }
                            }

                            output[get_index(output_shape, {0, i, j, k})] = move(multiplied_result);
                        }
                    }
                }
            }
        }

        return output;
    } else {
        exit(0);
    }

    return vector<variant<vector<double>, Ciphertext>>();
}

vector<variant<vector<double>, Ciphertext>> avg_pool(
    string dataset,
    SEALPACK &seal,
    MNIST_Shape shapes,
    int round,
    vector<variant<vector<double>, Ciphertext>> &input) {
    if (dataset == string("MNIST")) {
        auto input_shape = shapes.pool_input[round];
        auto output_shape = shapes.pool_output[round];
        size_t output_size = 1;
        for (size_t i = 1; i < output_shape.size(); ++i) {
            output_size *= output_shape[i];
        }
        vector<variant<vector<double>, Ciphertext>> output(output_size);

        SEALPACK seal;

        Plaintext quarter_plain;
        seal.encoder_.encode(0.25, SCALE, quarter_plain);

        const size_t batch_size = output_shape[0];

        for (size_t i = 0; i < output_shape[1]; ++i) {
            for (size_t j = 0; j < output_shape[2]; ++j) {
                for (size_t k = 0; k < output_shape[3]; ++k) {
                    if (get_if<Ciphertext>(&input[get_index(input_shape, {0, i, j * 2, k * 2})]) !=
                        nullptr) {
                        auto cipher_one =
                            get<Ciphertext>(input[get_index(input_shape, {0, i, j * 2, k * 2})]);
                        auto cipher_two = get<Ciphertext>(
                            input[get_index(input_shape, {0, i, j * 2 + 1, k * 2})]);
                        auto cipher_three = get<Ciphertext>(
                            input[get_index(input_shape, {0, i, j * 2, k * 2 + 1})]);
                        auto cipher_four = get<Ciphertext>(
                            input[get_index(input_shape, {0, i, j * 2 + 1, k * 2 + 1})]);

                        seal.evaluator_.add_inplace(cipher_one, cipher_two);
                        seal.evaluator_.add_inplace(cipher_three, cipher_four);
                        seal.evaluator_.add_inplace(cipher_one, cipher_three);

                        parms_id_type last_parms_id = cipher_one.parms_id();
                        seal.evaluator_.mod_switch_to_inplace(quarter_plain, last_parms_id);

                        seal.evaluator_.multiply_plain_inplace(cipher_one, quarter_plain);
                        seal.evaluator_.rescale_to_next_inplace(cipher_one);

                        output[get_index(output_shape, {0, i, j, k})] = cipher_one;
                    } else {
                        vector<double> values(batch_size);
                        for (size_t l = 0; l < batch_size; ++l) {
                            double pool_sum = 0;

                            pool_sum += get<vector<double>>(
                                input[get_index(input_shape, {0, i, j * 2, k * 2})])[l];
                            pool_sum += get<vector<double>>(
                                input[get_index(input_shape, {0, i, j * 2, k * 2 + 1})])[l];
                            pool_sum += get<vector<double>>(
                                input[get_index(input_shape, {0, i, j * 2 + 1, k * 2})])[l];
                            pool_sum += get<vector<double>>(
                                input[get_index(input_shape, {0, i, j * 2 + 1, k * 2 + 1})])[l];

                            double pool_average = pool_sum / 4;
                            values[l] = pool_average;
                        }

                        output[get_index(output_shape, {0, i, j, k})] = move(values);
                    }
                }
            }
        }

        return output;
    } else {
        return vector<variant<vector<double>, Ciphertext>>();
    }
}

vector<variant<vector<double>, Ciphertext>> recombine_input(
    const array<size_t, 4> input_shape,
    const double *input_data) {
    size_t block_size = 1;
    for (size_t i = 1; i < input_shape.size(); ++i) {
        block_size *= input_shape[i];
    }

    size_t batch_size = input_shape[0];
    vector<variant<vector<double>, Ciphertext>> result(block_size);
    for (size_t i = 0; i < block_size; ++i) {
        auto batch_values = vector<double>(batch_size);
        for (size_t j = 0; j < batch_size; ++j) {
            batch_values[j] = input_data[i + j * block_size];
        }

        result[i] = move(batch_values);
    }

    return result;
}

void save_worker_result(string dataset, vector<variant<vector<double>, Ciphertext>> &input) {
    // move all elements in the input
    const string result_path = DATA_PATH + string("communication/") + dataset + string("_result");

    ofstream result_output_stream;
    result_output_stream.open(result_path, ios::out | ios::binary);
    for (size_t i = 0; i < input.size(); ++i) {
        if (get_if<Ciphertext>(&input[i]) != nullptr) {
            auto cipher = move(get<Ciphertext>(input[i]));
            cipher.save(result_output_stream);
        } else {
            auto values = get<vector<double>>(input[i]);
            for (size_t j = 0; j < values.size(); ++j) {
                result_output_stream.write(
                    reinterpret_cast<const char *>(&values[j]), sizeof(double));
            }
        }
    }

    result_output_stream.close();
}

void square_activate(
    string dataset,
    SEALPACK &seal,
    vector<variant<vector<double>, Ciphertext>> &input) {
    if (dataset == string("MNIST")) {
        for (size_t i = 0; i < input.size(); ++i) {
            if (get_if<Ciphertext>(&input[i]) != nullptr) {
                Ciphertext input_cipher = get<Ciphertext>(input[i]);
                seal.evaluator_.square_inplace(input_cipher);
                seal.evaluator_.relinearize_inplace(input_cipher, seal.relin_keys_);
                seal.evaluator_.rescale_to_next_inplace(input_cipher);
                input[i] = input_cipher;
            } else {
                vector<double> values = get<vector<double>>(input[i]);
                for (size_t j = 0; j < values.size(); ++j) {
                    values[j] = values[j] * values[j];
                }

                input[i] = values;
            }
        }
    } else {
        exit(1);
    }
}

void worker(char *dataset, int batch_size, double *input_data, mode work_mode = separate_) {
    if (string(dataset) == string("MNIST")) {
        SEALPACK seal;

        auto shapes = get_MNIST_shapes(batch_size);
        auto encrypted_neurons = get_encrypted_neurons_list(dataset);
        auto input = recombine_input(shapes.conv_input[1], input_data);
        for (size_t round = 1; round <= shapes.conv_input.size(); ++round) {
            auto conv_result =
                conv(dataset, seal, shapes, round, input, encrypted_neurons, work_mode);
            auto avg_pool_result = avg_pool(dataset, seal, shapes, round, conv_result);
            square_activate(dataset, seal, avg_pool_result);
            input = move(avg_pool_result);
            if (round == 2) {
                cout << "Hello?" << endl;
                if (work_mode == separate_ or work_mode == remove_) {
                    for (size_t i = 0; i < input.size(); ++i) {
                        auto cipher = get<Ciphertext>(input[i]);
                        cout << log2(cipher.scale()) << endl;
                    }
                    save_worker_result(dataset, input);
                } else {
                }
            }
        }
    } else {
        exit(1);
    }
}
}