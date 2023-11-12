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
using namespace std;

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
    Plaintext plain_one;
    seal.encoder_.encode(one, SCALE, plain_one);

    for (size_t i = 0; i < level; ++i) {
        parms_id_type last_parms_id = cipher.parms_id();
        seal.evaluator_.mod_switch_to_inplace(plain_one, last_parms_id);
        seal.evaluator_.multiply_plain_inplace(cipher, plain_one);
        seal.evaluator_.rescale_to_next_inplace(cipher);
    }
}

vector<variant<double, Ciphertext>> read_conv_weight(
    string dataset,
    SEALPACK &seal,
    int round,
    json encrypted_neurons,
    mode work_mode) {
    auto shapes = Shapes[string(dataset)];
    auto weight_shape = shapes.conv_weight[round];
    size_t weight_size = 1;
    for (size_t i = 0; i < weight_shape.size(); ++i) {
        weight_size *= weight_shape[i];
    }
    vector<variant<double, Ciphertext>> weight(weight_size);

    hash<json> encrypted_list_hash;
    auto files_prefix = to_string(encrypted_list_hash(encrypted_neurons));
    string weight_path = move(
        DATA_PATH + string(dataset) + string("/") + files_prefix + string("_conv") +
        to_string(round) + string("_weight"));
    if (work_mode == full_) {
        weight_path += string("_full");
    }

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
    json encrypted_neurons,
    mode work_mode) {
    auto shape = Shapes[string(dataset)];
    hash<json> encrypted_list_hash;
    auto files_prefix = to_string(encrypted_list_hash(encrypted_neurons));
    string bias_path = move(
        DATA_PATH + string(dataset) + string("/") + files_prefix + string("_conv") +
        to_string(round) + string("_bias"));
    if (work_mode == full_) {
        bias_path += string("_full");
    }

    ifstream bias_instream;
    bias_instream.open(bias_path, ios::in | ios::binary);
    vector<variant<double, Ciphertext>> bias(shape.conv_bias[round]);
    for (size_t i = 0; i < bias.size(); ++i) {
        if (is_neuron_encrypted(encrypted_neurons, round, i)) {
            seal.cipher_.load(seal.context_, bias_instream);
            degrade_cipher_levels(
                seal, seal.cipher_, 1 + (round - 1) * (dataset == "CIFAR10" ? 2 : 3));
            bias[i] = seal.cipher_;
        } else {
            double bias_value;
            bias_instream.read(reinterpret_cast<char *>(&bias_value), sizeof(bias_value));
            bias[i] = bias_value;
        }
    }

    bias_instream.close();
    return bias;
}

void thread_conv_worker(
    SEALPACK &seal,
    const string &dataset,
    const vector<array<size_t, 4>> &output_indexes,
    const vector<variant<vector<double>, Ciphertext>> &input,
    const vector<variant<double, Ciphertext>> &weight,
    Ciphertext bias,
    vector<variant<vector<double>, Ciphertext>> &result,
    Shape shape,
    size_t round,
    int begining,
    int ending) {
    auto encrypted_neurons = get_encrypted_neurons_list(dataset);
    for (int i = begining; i < ending; ++i) {
        auto sum = bias;

        if (round == 1) {
            for (size_t channel = 0; channel < shape.conv_input[1][1]; ++channel) {
                for (size_t a = 0; a < shape.kernal_sizes[1]; ++a) {
                    for (size_t b = 0; b < shape.kernal_sizes[1]; ++b) {
                        if (dataset == "MNIST" || dataset == "EMNIST") {
                            auto input_values = get<vector<double>>(input[get_index(
                                shape.conv_input[1],
                                {0, channel, output_indexes[i][2] + a, output_indexes[i][3] + b})]);
                            Plaintext input_plain;
                            seal.encoder_.encode(input_values, SCALE, input_plain);
                            auto weight_cipher = get<Ciphertext>(weight[get_index(
                                shape.conv_weight[1], {output_indexes[i][1], channel, a, b})]);

                            Ciphertext multiplied_cipher;
                            seal.evaluator_.multiply_plain(
                                weight_cipher, input_plain, multiplied_cipher);
                            seal.evaluator_.rescale_to_next_inplace(multiplied_cipher);
                            seal.evaluator_.add_inplace(sum, multiplied_cipher);
                        } else {
                            if ((0 <= output_indexes[i][2] + a - 1 &&
                                 output_indexes[i][2] + a - 1 < shape.conv_input[1][2]) &&
                                (0 <= output_indexes[i][3] + b - 1 &&
                                 output_indexes[i][3] + b - 1 < shape.conv_input[1][3])) {
                                auto input_values = get<vector<double>>(input[get_index(
                                    shape.conv_input[1],
                                    {0,
                                     channel,
                                     output_indexes[i][2] + a - 1,
                                     output_indexes[i][3] + b - 1})]);
                                Plaintext input_plain;
                                seal.encoder_.encode(input_values, SCALE, input_plain);
                                auto weight_cipher = get<Ciphertext>(weight[get_index(
                                    shape.conv_weight[1], {output_indexes[i][1], channel, a, b})]);

                                Ciphertext multiplied_cipher;
                                seal.evaluator_.multiply_plain(
                                    weight_cipher, input_plain, multiplied_cipher);
                                seal.evaluator_.rescale_to_next_inplace(multiplied_cipher);
                                seal.evaluator_.add_inplace(sum, multiplied_cipher);
                            }
                        }
                    }
                }
            }

            result[get_index(
                shape.conv_output[1],
                {0, output_indexes[i][1], output_indexes[i][2], output_indexes[i][3]})] = move(sum);
        } else {
            for (size_t channel = 0; channel < shape.conv_input[round][1]; ++channel) {
                if (!is_neuron_encrypted(encrypted_neurons, round - 1, channel)) {
                    for (size_t a = 0; a < shape.kernal_sizes[2]; ++a) {
                        for (size_t b = 0; b < shape.kernal_sizes[2]; ++b) {
                            if (dataset == "MNIST" || dataset == "EMNIST") {
                                auto input_values = get<vector<double>>(input[get_index(
                                    shape.conv_input[round],
                                    {0,
                                     channel,
                                     output_indexes[i][2] + a,
                                     output_indexes[i][3] + b})]);
                                Plaintext input_plain;
                                seal.encoder_.encode(input_values, SCALE, input_plain);

                                auto weight_cipher = get<Ciphertext>(weight[get_index(
                                    shape.conv_weight[round],
                                    {output_indexes[i][1], channel, a, b})]);
                                seal.evaluator_.multiply_plain_inplace(weight_cipher, input_plain);
                                seal.evaluator_.rescale_to_next_inplace(weight_cipher);
                                degrade_cipher_levels(seal, weight_cipher, 3);
                                sum.scale() = weight_cipher.scale() = SCALE;
                                seal.evaluator_.add_inplace(sum, weight_cipher);
                            } else {
                                if ((0 <= output_indexes[i][2] + a - 1 &&
                                     output_indexes[i][2] + a - 1 < shape.conv_input[2][2]) &&
                                    (0 <= output_indexes[i][3] + b - 1 &&
                                     output_indexes[i][3] + b - 1 < shape.conv_input[2][3])) {
                                    auto input_values = get<vector<double>>(input[get_index(
                                        shape.conv_input[round],
                                        {0,
                                         channel,
                                         output_indexes[i][2] + a - 1,
                                         output_indexes[i][3] + b - 1})]);
                                    Plaintext input_plain;
                                    seal.encoder_.encode(input_values, SCALE, input_plain);
                                    auto weight_cipher = get<Ciphertext>(weight[get_index(
                                        shape.conv_weight[round],
                                        {output_indexes[i][1], channel, a, b})]);

                                    seal.evaluator_.multiply_plain_inplace(
                                        weight_cipher, input_plain);
                                    seal.evaluator_.rescale_to_next_inplace(weight_cipher);
                                    degrade_cipher_levels(
                                        seal, weight_cipher, dataset == "CIFAR10" ? 2 : 3);
                                    sum.scale() = weight_cipher.scale() = SCALE;
                                    seal.evaluator_.add_inplace(sum, weight_cipher);
                                }
                            }
                        }
                    }
                } else {
                    for (size_t a = 0; a < shape.kernal_sizes[2]; ++a) {
                        for (size_t b = 0; b < shape.kernal_sizes[2]; ++b) {
                            if (dataset == string("MNIST") || dataset == string("EMNIST")) {
                                auto input_cipher = get<Ciphertext>(input[get_index(
                                    shape.conv_input[round],
                                    {0,
                                     channel,
                                     output_indexes[i][2] + a,
                                     output_indexes[i][3] + b})]);

                                auto weight_cipher = get<Ciphertext>(weight[get_index(
                                    shape.conv_weight[round],
                                    {output_indexes[i][1], channel, a, b})]);

                                degrade_cipher_levels(seal, weight_cipher, 3);

                                seal.evaluator_.multiply_inplace(weight_cipher, input_cipher);
                                seal.evaluator_.relinearize_inplace(
                                    weight_cipher, seal.relin_keys_);
                                seal.evaluator_.rescale_to_next_inplace(weight_cipher);

                                sum.scale() = weight_cipher.scale() = SCALE;
                                seal.evaluator_.add_inplace(sum, weight_cipher);
                            } else {
                                if ((0 <= output_indexes[i][2] + a - 1 &&
                                     output_indexes[i][2] + a - 1 < shape.conv_input[2][2]) &&
                                    (0 <= output_indexes[i][3] + b - 1 &&
                                     output_indexes[i][3] + b - 1 < shape.conv_input[2][3])) {

                                    auto input_cipher = get<Ciphertext>(input[get_index(
                                        shape.conv_input[round],
                                        {0,
                                         channel,
                                         output_indexes[i][2] + a - 1,
                                         output_indexes[i][3] + b - 1})]);

                                    auto weight_cipher = get<Ciphertext>(weight[get_index(
                                        shape.conv_weight[round],
                                        {output_indexes[i][1], channel, a, b})]);

                                    degrade_cipher_levels(
                                        seal, weight_cipher, dataset == "CIFAR10" ? 2 : 3);
                                    seal.evaluator_.multiply_inplace(weight_cipher, input_cipher);

                                    seal.evaluator_.relinearize_inplace(
                                        weight_cipher, seal.relin_keys_);
                                    seal.evaluator_.rescale_to_next_inplace(weight_cipher);

                                    sum.scale() = weight_cipher.scale() = SCALE;
                                    seal.evaluator_.add_inplace(sum, weight_cipher);
                                }
                            }
                        }
                    }
                }
            }
            result[get_index(
                shape.conv_output[round],
                {0, output_indexes[i][1], output_indexes[i][2], output_indexes[i][3]})] = move(sum);
        }
    }
}

void thread_conv(
    SEALPACK &seal,
    const string &dataset,
    const vector<array<size_t, 4>> &output_indexes,
    const vector<variant<vector<double>, Ciphertext>> &input,
    const vector<variant<double, Ciphertext>> &weight,
    Ciphertext bias,
    vector<variant<vector<double>, Ciphertext>> &result,
    Shape shape,
    size_t round) {
    size_t processor_count = thread::hardware_concurrency();
    vector<size_t> threads_task_count(processor_count, output_indexes.size() / processor_count);
    threads_task_count[0] += output_indexes.size() % processor_count;

    int beginnig = 0;
    vector<thread> threads(processor_count);
    for (size_t i = 0; i < processor_count; ++i) {
        threads[i] = thread(
            thread_conv_worker,
            ref(seal),
            cref(dataset),
            cref(output_indexes),
            cref(input),
            cref(weight),
            bias,
            ref(result),
            shape,
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
    Shape shapes,
    int round,
    vector<variant<vector<double>, Ciphertext>> &input,
    json encrypted_neurons,
    mode work_mode) {
    auto conv_weight = read_conv_weight(dataset, seal, round, encrypted_neurons, work_mode);
    auto conv_bias = read_conv_bias(dataset, seal, round, encrypted_neurons, work_mode);
    size_t kernal_size = shapes.kernal_sizes[round];
    array<size_t, 4> output_shape = shapes.conv_output[round];
    const size_t batch_size = output_shape[0];
    size_t output_size = 1;
    for (size_t i = 1; i < output_shape.size(); ++i) {
        output_size *= output_shape[i];
    }
    vector<variant<vector<double>, Ciphertext>> output(output_size);
    if (round == 1) { // conv1 inputs are plain
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
                    seal, dataset, work_indexes, input, conv_weight, bias, output, shapes, round);
            } else {
                for (size_t j = 0; j < output_shape[2]; ++j) {
                    for (size_t k = 0; k < output_shape[3]; ++k) {
                        double bias = get<double>(conv_bias[i]);
                        vector<double> multiplied_result(batch_size, bias);
                        for (size_t channel = 0; channel < shapes.conv_input[round][1]; ++channel) {
                            if (dataset == "MNIST" || dataset == "EMNIST") {
                                for (size_t a = 0; a < kernal_size; ++a) {
                                    for (size_t b = 0; b < kernal_size; ++b) {
                                        auto input_values = get<vector<double>>(input[get_index(
                                            shapes.conv_input[1], {0, channel, j + a, k + b})]);
                                        double weight_value = get<double>(conv_weight[get_index(
                                            shapes.conv_weight[1], {i, channel, a, b})]);
                                        for (size_t l = 0; l < batch_size; ++l) {
                                            multiplied_result[l] += input_values[l] * weight_value;
                                        }
                                    }
                                }
                            } else {
                                for (size_t a = 0; a < kernal_size; ++a) {
                                    for (size_t b = 0; b < kernal_size; ++b) {
                                        if ((0 <= j + a - 1 &&
                                             j + a - 1 < shapes.conv_input[1][2]) &&
                                            (0 <= k + b - 1 &&
                                             k + b - 1 < shapes.conv_input[1][3])) {
                                            auto input_values = get<vector<double>>(input[get_index(
                                                shapes.conv_input[1],
                                                {0, channel, j + a - 1, k + b - 1})]);
                                            double weight_value = get<double>(conv_weight[get_index(
                                                shapes.conv_weight[1], {i, channel, a, b})]);
                                            for (size_t l = 0; l < batch_size; ++l) {
                                                multiplied_result[l] +=
                                                    input_values[l] * weight_value;
                                            }
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
    } else { // later conv inputs are plain/cipher
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
                    seal, dataset, work_indexes, input, conv_weight, bias, output, shapes, round);
            } else {
                for (size_t j = 0; j < output_shape[2]; ++j) {
                    for (size_t k = 0; k < output_shape[3]; ++k) {
                        double bias = get<double>(conv_bias[i]);
                        vector<double> multiplied_result(batch_size, bias);
                        for (size_t channel = 0; channel < shapes.conv_output[1][1]; ++channel) {
                            if (!is_neuron_encrypted(encrypted_neurons, round - 1, channel)) {
                                for (size_t a = 0; a < kernal_size; ++a) {
                                    for (size_t b = 0; b < kernal_size; ++b) {
                                        if (dataset == string("MNIST") ||
                                            dataset == string("EMNIST")) {
                                            auto input_values = get<vector<double>>(input[get_index(
                                                shapes.conv_input[round],
                                                {0, channel, j + a, k + b})]);
                                            double weight_value = get<double>(conv_weight[get_index(
                                                shapes.conv_weight[round], {i, channel, a, b})]);
                                            for (size_t l = 0; l < batch_size; ++l) {
                                                multiplied_result[l] +=
                                                    input_values[l] * weight_value;
                                            }
                                        } else {
                                            if ((0 <= j + a - 1 &&
                                                 j + a - 1 < shapes.conv_input[round][2]) &&
                                                (0 <= k + b - 1 &&
                                                 k + b - 1 < shapes.conv_input[round][3])) {
                                                auto input_values =
                                                    get<vector<double>>(input[get_index(
                                                        shapes.conv_input[round],
                                                        {0, channel, j + a - 1, k + b - 1})]);
                                                double weight_value =
                                                    get<double>(conv_weight[get_index(
                                                        shapes.conv_weight[round],
                                                        {i, channel, a, b})]);
                                                for (size_t l = 0; l < batch_size; ++l) {
                                                    multiplied_result[l] +=
                                                        input_values[l] * weight_value;
                                                }
                                            }
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
}

vector<variant<vector<double>, Ciphertext>> avg_pool(
    string dataset,
    SEALPACK &seal,
    Shape shapes,
    int round,
    vector<variant<vector<double>, Ciphertext>> &input) {

    auto input_shape = shapes.pool_input[round];
    auto output_shape = shapes.pool_output[round];
    size_t output_size = 1;
    for (size_t i = 1; i < output_shape.size(); ++i) {
        output_size *= output_shape[i];
    }
    vector<variant<vector<double>, Ciphertext>> output(output_size);

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
                    auto cipher_two =
                        get<Ciphertext>(input[get_index(input_shape, {0, i, j * 2 + 1, k * 2})]);
                    auto cipher_three =
                        get<Ciphertext>(input[get_index(input_shape, {0, i, j * 2, k * 2 + 1})]);
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
}

vector<variant<vector<double>, Ciphertext>> recombine_input(
    const array<size_t, 4> input_shape,
    const double *input_data) {
    size_t batch_size = input_shape[0];
    size_t block_size = input_shape[1] * input_shape[2] * input_shape[3];
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
    // move all input elements!
    const string result_path =
        DATA_PATH + string("communication/") + dataset + string("_conv2_result");

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
}

void square_activate_fc(string dataset, SEALPACK &seal, vector<Ciphertext> &input) {
    if (dataset == string("MNIST")) {
        for (size_t i = 0; i < input.size(); ++i) {
            auto input_cipher = input[i];
            seal.evaluator_.square_inplace(input_cipher);
            seal.evaluator_.relinearize_inplace(input_cipher, seal.relin_keys_);
            seal.evaluator_.rescale_to_next_inplace(input_cipher);
            input[i] = input_cipher;
        }
    } else {
        exit(1);
    }
}

vector<size_t> read_fc_poi(SEALPACK &seal, const string &path, array<size_t, 2> weight_shape) {
    vector<size_t> pois{0};
    ifstream fc_instream;
    fc_instream.open(path, ios::in | ios::binary);
    for (size_t i = 0; i < weight_shape[1]; ++i) {
        for (size_t j = 0; j < weight_shape[0]; ++j) {
            seal.cipher_.load(seal.context_, fc_instream);
        }
        pois.push_back(fc_instream.tellg());
    }

    return pois;
}

vector<Ciphertext> read_fc_weight(SEALPACK &seal, const string &path, size_t poi, size_t size) {
    ifstream fc_instream;
    fc_instream.open(path, ios::in | ios::binary);
    fc_instream.seekg(poi, fc_instream.beg);
    vector<Ciphertext> fc_weight;
    Ciphertext cipher;
    for (size_t i = 0; i < size; ++i) {
        cipher.load(seal.context_, fc_instream);
        fc_weight.emplace_back(cipher);
    }
    fc_instream.close();

    return fc_weight;
}

vector<Ciphertext> read_fc_bias(SEALPACK &seal, const string &path, size_t size) {
    ifstream fc_instream;
    fc_instream.open(path, ios::in | ios::binary);

    vector<Ciphertext> fc_data;
    for (size_t i = 0; i < size; ++i) {
        seal.cipher_.load(seal.context_, fc_instream);
        fc_data.emplace_back(seal.cipher_);
    }
    fc_instream.close();

    return fc_data;
}

void save_fc_result(string dataset, vector<Ciphertext> &input) {
    // move all input elements!
    const string result_path =
        DATA_PATH + string("communication/") + dataset + string("_fc3_result");

    ofstream result_output_stream;
    result_output_stream.open(result_path, ios::out | ios::binary);
    for (size_t i = 0; i < input.size(); ++i) {
        auto cipher = move(input[i]);
        cipher.save(result_output_stream);
    }

    result_output_stream.close();
}

vector<Ciphertext> read_full_cipher_result(SEALPACK &seal, string name, size_t size) {
    const string result_path = DATA_PATH + string("communication/MNIST_") + name;
    ifstream data_input_stream;
    data_input_stream.open(result_path, ios::in | ios::binary);
    vector<Ciphertext> result;
    for (size_t i = 0; i < size; ++i) {
        seal.cipher_.load(seal.context_, data_input_stream);
        result.emplace_back(seal.cipher_);
    }

    data_input_stream.close();
    return result;
}

void save_full_cipher_result(string name, vector<Ciphertext> &input) {
    const string result_path = DATA_PATH + string("communication/MNIST_") + name;

    ofstream result_output_stream;
    result_output_stream.open(result_path, ios::out | ios::binary);
    for (size_t i = 0; i < input.size(); ++i) {
        auto cipher = move(input[i]);
        cipher.save(result_output_stream);
    }

    result_output_stream.close();
}

void fc_worker(
    SEALPACK &seal,
    int round,
    const string &path,
    const vector<Ciphertext> &input,
    const vector<size_t> &weight_poi,
    const vector<Ciphertext> &bias_data,
    vector<Ciphertext> &result,
    int beginning,
    int ending) {
    auto shapes = Shapes[(string("MNIST"))];
    auto weight_shape = shapes.fc_weight[round];
    for (int index = beginning; index < ending; ++index) {
        auto sum = bias_data[index];
        degrade_cipher_levels(seal, sum, 5 + round * 2);
        auto weight = read_fc_weight(seal, path, weight_poi[index], weight_shape[0]);
        for (size_t j = 0; j < weight_shape[0]; ++j) {
            degrade_cipher_levels(seal, weight[j], 5 + round * 2 - 1);

            seal.evaluator_.multiply_inplace(weight[j], input[j]);
            seal.evaluator_.relinearize_inplace(weight[j], seal.relin_keys_);
            seal.evaluator_.rescale_to_next_inplace(weight[j]);

            sum.scale() = weight[j].scale() = SCALE;
            seal.evaluator_.add_inplace(sum, weight[j]);
        }
        vector<Ciphertext>().swap(weight);
        result[index] = move(sum);
    }
}

vector<Ciphertext> fc(SEALPACK &seal, const vector<Ciphertext> &input, size_t round) {
    const string MNIST_path = DATA_PATH + string("MNIST/");
    auto encryped_neurons = get_encrypted_neurons_list("MNIST");
    hash<json> encrypted_list_hash;
    auto files_prefix = to_string(encrypted_list_hash(encryped_neurons));
    const string fc_weight_path =
        MNIST_path + files_prefix + string("_fc") + to_string(round) + string("_weight_full");
    const string fc_bias_path =
        MNIST_path + files_prefix + string("_fc") + to_string(round) + string("_bias_full");

    auto shapes = Shapes[(string("MNIST"))];
    auto weight_shape = shapes.fc_weight[round];

    auto bias_data = read_fc_bias(seal, fc_bias_path, weight_shape[1]);
    auto weight_pois = read_fc_poi(seal, fc_weight_path, weight_shape);

    vector<Ciphertext> result(weight_shape[1]);
    const size_t processor_count = thread::hardware_concurrency();
    vector<size_t> threads_task_count(processor_count, weight_shape[1] / processor_count);
    threads_task_count[0] += weight_shape[1] % processor_count;

    int beginnig = 0;
    vector<thread> threads(processor_count);
    for (size_t i = 0; i < processor_count; ++i) {
        threads[i] = thread(
            fc_worker,
            ref(seal),
            round,
            cref(fc_weight_path),
            cref(input),
            cref(weight_pois),
            cref(bias_data),
            ref(result),
            beginnig,
            beginnig + threads_task_count[i]);
        beginnig += threads_task_count[i];
    }

    for (size_t i = 0; i < threads.size(); ++i) {
        threads[i].join();
    }

    return result;
}

void worker(const char *dataset, int batch_size, double *input_data, mode work_mode = separate_) {
    SEALPACK seal(work_mode);
    auto shape = Shapes[string(dataset)];

    update_shape_size(shape, batch_size);
    auto encrypted_neurons = get_encrypted_neurons_list(dataset);
    auto input = recombine_input(shape.conv_input[1], input_data);

    for (size_t round = 1; round <= shape.conv_input.size(); ++round) {
        auto conv_result = conv(dataset, seal, shape, round, input, encrypted_neurons, work_mode);

        if (round == 2) {
            if (work_mode == separate_ or work_mode == remove_) {
                save_worker_result(dataset, conv_result);
            } else {
                auto avg_pool_result = avg_pool(dataset, seal, shape, round, conv_result);
                vector<variant<vector<double>, Ciphertext>>().swap(conv_result);
                square_activate(dataset, seal, avg_pool_result);
                vector<Ciphertext> fc_input;

                for (size_t i = 0; i < avg_pool_result.size(); ++i) {
                    fc_input.emplace_back(get<Ciphertext>(avg_pool_result[i]));
                }
                vector<variant<vector<double>, Ciphertext>>().swap(avg_pool_result);

                auto fc1_output = fc(seal, fc_input, 1);
                vector<seal::Ciphertext>().swap(fc_input);
                square_activate_fc(dataset, seal, fc1_output);

                auto fc2_output = fc(seal, fc1_output, 2);
                vector<seal::Ciphertext>().swap(fc1_output);
                square_activate_fc(dataset, seal, fc2_output);

                auto fc3_output = fc(seal, fc2_output, 3);
                vector<seal::Ciphertext>().swap(fc2_output);
                save_fc_result(dataset, fc3_output);
            }
        } else {
            if (string(dataset) == string("CIFAR10")) {
                square_activate(dataset, seal, conv_result);
                input = move(conv_result);
            } else {
                auto avg_pool_result = avg_pool(dataset, seal, shape, round, conv_result);
                vector<variant<vector<double>, Ciphertext>>().swap(conv_result);
                square_activate(dataset, seal, avg_pool_result);
                input = move(avg_pool_result);
            }
        }
    }
}

int main(int argc, char *argv[]) {
    cout << "Hello World!" << endl;
    return 0;
}
}