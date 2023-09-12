#include "common.h"
#include "timer.h"
#include "utils.h"

#include <array>
#include <fstream>
#include <memory>
#include <thread>
#include <typeinfo>
#include <variant>

extern "C" {

bool is_file_complete(char *dataset) {
    if (string(dataset) == string("MNIST")) {
        auto encrypted_neurons = get_encrypted_neurons_list(dataset);
        hash<json> encrypted_list_hash;
        auto files_prefix = to_string(encrypted_list_hash(encrypted_neurons));

        const string MNIST_path = DATA_PATH + string(dataset) + string("/");
        const array<string, 4> file_names{
            string("_conv1_weight"),
            string("_conv1_bias"),
            string("_conv2_weight"),
            string("_conv2_bias")};
        for (size_t i = 0; i < file_names.size(); ++i) {
            if (!is_file_exist(MNIST_path + files_prefix + file_names[i])) {
                return false;
            }
        }

        return true;
    } else {
        exit(1);
    }
}

void save_trained_data(char *dataset, double *trained_data) {
    save_parms();
    save_keys();
    if (string(dataset) == string("MNIST")) {
        MNIST_Shape shapes;
        const string MNIST_path = DATA_PATH + string(dataset) + string("/");

        auto parms = read_parms();
        auto secret_key = read_secret_key();
        SEALContext context(parms);
        CKKSEncoder encoder(context);
        Encryptor encryptor(context, secret_key);

        Plaintext temp_plain;
        Ciphertext temp_cipher;
        int trained_data_index = 0;
        auto encryped_neurons = get_encrypted_neurons_list(dataset);
        hash<json> encrypted_list_hash;
        auto files_prefix = to_string(encrypted_list_hash(encryped_neurons));

        // conv1_weight
        ofstream conv1_weight_outstream;
        const string conv1_weight_path = MNIST_path + files_prefix + string("_conv1_weight");
        conv1_weight_outstream.open(conv1_weight_path, ios::out | ios::binary);

        auto conv1_weight_shape = shapes.conv_weight[1];
        size_t conv1_weight_size = 1;
        for (size_t i = 1; i < conv1_weight_shape.size(); ++i) {
            conv1_weight_size *= conv1_weight_shape[i];
        }
        for (size_t i = 0; i < conv1_weight_shape[0]; ++i) {
            for (size_t j = 0; j < conv1_weight_size; ++j) {
                if (is_neuron_encrypted(encryped_neurons, 1, i)) {
                    encoder.encode(trained_data[trained_data_index], SCALE, temp_plain);
                    encryptor.encrypt_symmetric(temp_plain, temp_cipher);
                    temp_cipher.save(conv1_weight_outstream);
                } else {
                    double temp_number = trained_data[trained_data_index];
                    conv1_weight_outstream.write(
                        reinterpret_cast<const char *>(&temp_number), sizeof(temp_number));
                }

                ++trained_data_index;
            }
        }
        conv1_weight_outstream.close();

        // conv1_bias
        ofstream conv1_bias_outstream;
        const string conv1_bias_path = MNIST_path + files_prefix + string("_conv1_bias");
        conv1_bias_outstream.open(conv1_bias_path, ios::out | ios::binary);

        auto conv1_bias_shape = shapes.conv_bias[1];
        for (size_t i = 0; i < conv1_bias_shape; ++i) {
            if (is_neuron_encrypted(encryped_neurons, 1, i)) {
                encoder.encode(trained_data[trained_data_index], SCALE, temp_plain);
                encryptor.encrypt_symmetric(temp_plain, temp_cipher);
                temp_cipher.save(conv1_bias_outstream);
            } else {
                double temp_number = trained_data[trained_data_index];
                conv1_bias_outstream.write(
                    reinterpret_cast<const char *>(&temp_number), sizeof(temp_number));
            }

            ++trained_data_index;
        }
        conv1_bias_outstream.close();

        // conv2_weight
        ofstream conv2_weight_outstream;
        const string conv2_weight_path = MNIST_path + files_prefix + string("_conv2_weight");
        conv2_weight_outstream.open(conv2_weight_path, ios::out | ios::binary);

        auto conv2_weight_shape = shapes.conv_weight[2];
        size_t conv2_weight_size = 1;
        for (size_t i = 1; i < conv2_weight_shape.size(); ++i) {
            conv2_weight_size *= conv2_weight_shape[i];
        }
        for (size_t i = 0; i < conv2_weight_shape[0]; ++i) {
            for (size_t j = 0; j < conv2_weight_size; ++j) {
                if (is_neuron_encrypted(encryped_neurons, 2, i)) {
                    encoder.encode(trained_data[trained_data_index], SCALE, temp_plain);
                    encryptor.encrypt_symmetric(temp_plain, temp_cipher);
                    temp_cipher.save(conv2_weight_outstream);
                } else {
                    double trained_data_value = trained_data[trained_data_index];
                    conv2_weight_outstream.write(
                        reinterpret_cast<const char *>(&trained_data_value),
                        sizeof(trained_data_value));
                }

                ++trained_data_index;
            }
        }
        conv2_weight_outstream.close();

        // conv2_bias
        ofstream conv2_bias_outstream;
        const string conv2_bias_path = MNIST_path + files_prefix + string("_conv2_bias");
        conv2_bias_outstream.open(conv2_bias_path, ios::out | ios::binary);

        auto conv2_bias_shape = shapes.conv_bias[2];
        for (size_t i = 0; i < conv2_bias_shape; ++i) {
            if (is_neuron_encrypted(encryped_neurons, 2, i)) {
                encoder.encode(trained_data[trained_data_index], SCALE, temp_plain);
                encryptor.encrypt_symmetric(temp_plain, temp_cipher);
                temp_cipher.save(conv2_bias_outstream);
            } else {
                double temp_number = trained_data[trained_data_index];
                conv2_bias_outstream.write(
                    reinterpret_cast<const char *>(&temp_number), sizeof(temp_number));
            }

            ++trained_data_index;
        }
        conv2_bias_outstream.close();
        cout << __LINE__ << "|" << trained_data_index << endl;
        // fc1_weight
        ofstream fc1_weight_outstream;
        const string fc1_weight_path = MNIST_path + files_prefix + string("_fc1_weight");
        fc1_weight_outstream.open(fc1_weight_path, ios::out | ios::binary);

        auto fc1_weight_shape = shapes.fc_weight[1];
        size_t fc1_weight_size = fc1_weight_shape[0] * fc1_weight_shape[1];
        for (size_t i = 0; i < fc1_weight_size; ++i) {
            encoder.encode(trained_data[trained_data_index], SCALE, temp_plain);
            encryptor.encrypt_symmetric(temp_plain, temp_cipher);
            temp_cipher.save(fc1_weight_outstream);
            ++trained_data_index;
        }
        fc1_weight_outstream.close();

        // fc1_bias
        ofstream fc1_bias_outstream;
        const string fc1_bias_path = MNIST_path + files_prefix + string("_fc1_bias");
        fc1_bias_outstream.open(fc1_bias_path, ios::out | ios::binary);

        size_t fc1_bias_size = shapes.fc_bias[1];
        for (size_t i = 0; i < fc1_bias_size; ++i) {
            encoder.encode(trained_data[trained_data_index], SCALE, temp_plain);
            encryptor.encrypt_symmetric(temp_plain, temp_cipher);
            temp_cipher.save(fc1_bias_outstream);
            ++trained_data_index;
        }
        fc1_bias_outstream.close();

        // fc2_weight
        ofstream fc2_weight_outstream;
        const string fc2_weight_path = MNIST_path + files_prefix + string("_fc2_weight");
        fc2_weight_outstream.open(fc2_weight_path, ios::out | ios::binary);

        auto fc2_weight_shape = shapes.fc_weight[2];
        size_t fc2_weight_size = fc2_weight_shape[0] * fc2_weight_shape[1];
        for (size_t i = 0; i < fc2_weight_size; ++i) {
            encoder.encode(trained_data[trained_data_index], SCALE, temp_plain);
            encryptor.encrypt_symmetric(temp_plain, temp_cipher);
            temp_cipher.save(fc2_weight_outstream);
            ++trained_data_index;
        }
        fc2_weight_outstream.close();

        // fc2_bias
        ofstream fc2_bias_outstream;
        const string fc2_bias_path = MNIST_path + files_prefix + string("_fc2_bias");
        fc2_bias_outstream.open(fc2_bias_path, ios::out | ios::binary);

        size_t fc2_bias_size = shapes.fc_bias[2];
        for (size_t i = 0; i < fc2_bias_size; ++i) {
            encoder.encode(trained_data[trained_data_index], SCALE, temp_plain);
            encryptor.encrypt_symmetric(temp_plain, temp_cipher);
            temp_cipher.save(fc2_bias_outstream);
            ++trained_data_index;
        }
        fc2_bias_outstream.close();

        // fc3_weight
        ofstream fc3_weight_outstream;
        const string fc3_weight_path = MNIST_path + files_prefix + string("_fc3_weight");
        fc3_weight_outstream.open(fc3_weight_path, ios::out | ios::binary);

        auto fc3_weight_shape = shapes.fc_weight[3];
        size_t fc3_weight_size = fc3_weight_shape[0] * fc3_weight_shape[1];
        for (size_t i = 0; i < fc3_weight_size; ++i) {
            encoder.encode(trained_data[trained_data_index], SCALE, temp_plain);
            encryptor.encrypt_symmetric(temp_plain, temp_cipher);
            temp_cipher.save(fc3_weight_outstream);
            ++trained_data_index;
        }
        fc3_weight_outstream.close();
        cout << __LINE__ << "|" << trained_data_index << endl;
        // fc3_bias
        ofstream fc3_bias_outstream;
        const string fc3_bias_path = MNIST_path + files_prefix + string("_fc3_bias");
        fc3_bias_outstream.open(fc3_bias_path, ios::out | ios::binary);

        size_t fc3_bias_size = shapes.fc_bias[3];
        for (size_t i = 0; i < fc3_bias_size; ++i) {
            encoder.encode(trained_data[trained_data_index], SCALE, temp_plain);
            encryptor.encrypt_symmetric(temp_plain, temp_cipher);
            temp_cipher.save(fc3_bias_outstream);
            ++trained_data_index;
        }
        fc3_bias_outstream.close();
        cout << __LINE__ << "|" << trained_data_index << endl;
    } else {
        exit(1);
    }
}

void thread_worker_decrypt_result(
    const vector<Ciphertext> &ciphers,
    vector<double> &numbers,
    int begining,
    int ending) {
    auto parms = read_parms();
    auto secret_key = read_secret_key();
    SEALContext context(parms);
    CKKSEncoder encoder(context);
    Decryptor decryptor(context, secret_key);
    Plaintext plaintext;
    vector<double> decrypt_results;

    for (int i = begining; i < ending; ++i) {
        decryptor.decrypt(ciphers[i], plaintext);
        encoder.decode(plaintext, decrypt_results);
        numbers[i] = decrypt_results[0];
    }
}

void thread_decrypt_result(const vector<Ciphertext> &ciphers, vector<double> &numbers) {
    const unsigned processor_count = std::thread::hardware_concurrency();
    vector<int> threads_task_count(processor_count, ciphers.size() / processor_count);
    threads_task_count[0] += ciphers.size() % processor_count;

    int beginnig = 0;
    vector<thread> threads(processor_count);
    for (size_t i = 0; i < threads.size(); ++i) {
        threads[i] = thread(
            thread_worker_decrypt_result,
            cref(ciphers),
            ref(numbers),
            beginnig,
            beginnig + threads_task_count[i]);
        beginnig += threads_task_count[i];
    }

    for (size_t i = 0; i < threads.size(); ++i) {
        threads[i].join();
    }
}

vector<double> read_result(
    const string &dataset,
    SEALPACK &seal,
    array<size_t, 4> output_shape,
    size_t output_size,
    mode work_mode) {
    if (dataset == string("MNIST")) {
        const string result_path =
            DATA_PATH + string("communication/") + dataset + string("_result");

        ifstream result_stream;
        result_stream.open(result_path, ios::in | ios::binary);
        vector<double> output(output_size);
        vector<double> decrypt_results(output_size);

        auto encrypted_neurons = get_encrypted_neurons_list("MNIST");
        const size_t batch_size = output_shape[0];
        size_t block_size = output_shape[2] * output_shape[3];
        for (size_t i = 0; i < output_size / output_shape[0];) {
            if (is_neuron_encrypted(encrypted_neurons, 2, i / block_size) &&
                work_mode == separate_) {
                for (size_t j = 0; j < block_size; ++i, ++j) {
                    seal.cipher_.load(seal.context_, result_stream);
                    seal.decryptor_.decrypt(seal.cipher_, seal.plain_);
                    seal.encoder_.decode(seal.plain_, decrypt_results);
                    for (size_t k = 0; k < batch_size; ++k) {
                        output[i + k * output_size / batch_size] = decrypt_results[k];
                    }
                }
            } else {
                for (size_t j = 0; j < block_size; ++i, ++j) {
                    double value;
                    for (size_t k = 0; k < batch_size; ++k) {
                        result_stream.read(reinterpret_cast<char *>(&value), sizeof(double));
                        output[i + k * output_size / batch_size] = value;
                    }
                }
            }
        }

        result_stream.close();
        return output;
    } else {
        exit(1);
    }
}

double *get_result(char *dataset, int batch_size, mode work_mode = separate_) {
    SEALPACK seal;
    auto shapes = get_MNIST_shapes(batch_size);
    auto output_shape = shapes.pool_output[2];
    size_t output_size = 1;
    for (size_t i = 0; i < output_shape.size(); ++i) {
        output_size *= output_shape[i];
    }

    auto final_result = read_result(string(dataset), seal, output_shape, output_size, work_mode);
    double *output = new double[output_size];
    for (size_t i = 0; i < output_size; ++i) {
        output[i] = final_result[i];
    }

    return output;
}
}