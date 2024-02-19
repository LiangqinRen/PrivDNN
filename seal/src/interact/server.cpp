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

bool is_file_complete(char *dataset, mode work_mode) {
    auto encrypted_neurons = get_encrypted_neurons_list(dataset);
    hash<json> encrypted_list_hash;
    auto files_prefix = to_string(encrypted_list_hash(encrypted_neurons));

    const string path = DATA_PATH + string(dataset) + string("/");
    vector<string> file_names;
    if (work_mode == separate_ || work_mode == remove_) {
        file_names = {
            string("_conv1_weight"),
            string("_conv1_bias"),
            string("_conv2_weight"),
            string("_conv2_bias")};
    } else {
        file_names = {
            string("_conv1_weight_full"),
            string("_conv1_bias_full"),
            string("_conv2_weight_full"),
            string("_conv2_bias_full"),
            string("_fc1_weight_full"),
            string("_fc1_bias_full"),
            string("_fc2_weight_full"),
            string("_fc2_bias_full"),
            string("_fc3_weight_full"),
            string("_fc3_bias_full")};
    }

    file_names.emplace_back("_bn1_weight");
    file_names.emplace_back("_bn1_bias");

    for (size_t i = 0; i < file_names.size(); ++i) {
        if (!is_file_exist(path + files_prefix + file_names[i])) {
            return false;
        }
    }

    return true;
}

void save_trained_data(char *dataset, double *trained_data, mode work_mode) {
    save_parms(work_mode);
    save_keys(work_mode);

    auto shape = Shapes[string(dataset)];
    const string dataset_path = DATA_PATH + string(dataset) + string("/");

    auto parms = read_parms(work_mode);
    auto secret_key = read_secret_key(work_mode);
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
    string conv1_weight_path = dataset_path + files_prefix + string("_conv1_weight");
    if (work_mode == full_) {
        conv1_weight_path += string("_full");
    }
    conv1_weight_outstream.open(conv1_weight_path, ios::out | ios::binary);

    auto conv1_weight_shape = shape.conv_weight[1];
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
    string conv1_bias_path = dataset_path + files_prefix + string("_conv1_bias");
    if (work_mode == full_) {
        conv1_bias_path += string("_full");
    }
    conv1_bias_outstream.open(conv1_bias_path, ios::out | ios::binary);

    auto conv1_bias_shape = shape.conv_bias[1];
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

    //  conv2_weight
    ofstream conv2_weight_outstream;
    string conv2_weight_path = dataset_path + files_prefix + string("_conv2_weight");
    if (work_mode == full_) {
        conv2_weight_path += string("_full");
    }
    conv2_weight_outstream.open(conv2_weight_path, ios::out | ios::binary);

    auto conv2_weight_shape = shape.conv_weight[2];
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

    //  conv2_bias
    ofstream conv2_bias_outstream;
    string conv2_bias_path = dataset_path + files_prefix + string("_conv2_bias");
    if (work_mode == full_) {
        conv2_bias_path += string("_full");
    }
    conv2_bias_outstream.open(conv2_bias_path, ios::out | ios::binary);

    auto conv2_bias_shape = shape.conv_bias[2];
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

    // bn1_weight
    ofstream bn1_weight_outstream;
    string bn1_weight_path = dataset_path + files_prefix + string("_bn1_weight");
    if (work_mode == full_) {
        bn1_weight_path += string("_full");
    }
    bn1_weight_outstream.open(bn1_weight_path, ios::out | ios::binary);

    auto bn1_weight_shape = shape.conv_bias[1];
    for (size_t i = 0; i < bn1_weight_shape; ++i) {
        if (is_neuron_encrypted(encryped_neurons, 1, i)) {
            encoder.encode(trained_data[trained_data_index], SCALE, temp_plain);
            encryptor.encrypt_symmetric(temp_plain, temp_cipher);
            temp_cipher.save(bn1_weight_outstream);
        } else {
            double temp_number = trained_data[trained_data_index];
            bn1_weight_outstream.write(
                reinterpret_cast<const char *>(&temp_number), sizeof(temp_number));
        }

        ++trained_data_index;
    }
    bn1_weight_outstream.close();

    // bn1_bias
    ofstream bn1_bias_outstream;
    string bn1_bias_path = dataset_path + files_prefix + string("_bn1_bias");
    if (work_mode == full_) {
        bn1_bias_path += string("_full");
    }
    bn1_bias_outstream.open(bn1_bias_path, ios::out | ios::binary);

    auto bn1_bias_shape = shape.conv_bias[1];
    for (size_t i = 0; i < bn1_bias_shape; ++i) {
        if (is_neuron_encrypted(encryped_neurons, 1, i)) {
            encoder.encode(trained_data[trained_data_index], SCALE, temp_plain);
            encryptor.encrypt_symmetric(temp_plain, temp_cipher);
            temp_cipher.save(bn1_bias_outstream);
        } else {
            double temp_number = trained_data[trained_data_index];
            bn1_bias_outstream.write(
                reinterpret_cast<const char *>(&temp_number), sizeof(temp_number));
        }

        ++trained_data_index;
    }
    bn1_bias_outstream.close();

    if (work_mode == full_) {
        // fc1_weight
        ofstream fc1_weight_outstream;
        string fc1_weight_path = dataset_path + files_prefix + string("_fc1_weight");
        if (work_mode == full_) {
            fc1_weight_path += string("_full");
        }
        fc1_weight_outstream.open(fc1_weight_path, ios::out | ios::binary);

        auto fc1_weight_shape = shape.fc_weight[1];
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
        string fc1_bias_path = dataset_path + files_prefix + string("_fc1_bias");
        if (work_mode == full_) {
            fc1_bias_path += string("_full");
        }
        fc1_bias_outstream.open(fc1_bias_path, ios::out | ios::binary);

        size_t fc1_bias_size = shape.fc_bias[1];
        for (size_t i = 0; i < fc1_bias_size; ++i) {
            encoder.encode(trained_data[trained_data_index], SCALE, temp_plain);
            encryptor.encrypt_symmetric(temp_plain, temp_cipher);
            temp_cipher.save(fc1_bias_outstream);
            ++trained_data_index;
        }
        fc1_bias_outstream.close();

        // fc2_weight
        ofstream fc2_weight_outstream;
        string fc2_weight_path = dataset_path + files_prefix + string("_fc2_weight");
        if (work_mode == full_) {
            fc2_weight_path += string("_full");
        }
        fc2_weight_outstream.open(fc2_weight_path, ios::out | ios::binary);

        auto fc2_weight_shape = shape.fc_weight[2];
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
        string fc2_bias_path = dataset_path + files_prefix + string("_fc2_bias");
        if (work_mode == full_) {
            fc2_bias_path += string("_full");
        }
        fc2_bias_outstream.open(fc2_bias_path, ios::out | ios::binary);

        size_t fc2_bias_size = shape.fc_bias[2];
        for (size_t i = 0; i < fc2_bias_size; ++i) {
            encoder.encode(trained_data[trained_data_index], SCALE, temp_plain);
            encryptor.encrypt_symmetric(temp_plain, temp_cipher);
            temp_cipher.save(fc2_bias_outstream);
            ++trained_data_index;
        }
        fc2_bias_outstream.close();

        // fc3_weight
        ofstream fc3_weight_outstream;
        string fc3_weight_path = dataset_path + files_prefix + string("_fc3_weight");
        if (work_mode == full_) {
            fc3_weight_path += string("_full");
        }
        fc3_weight_outstream.open(fc3_weight_path, ios::out | ios::binary);

        auto fc3_weight_shape = shape.fc_weight[3];
        size_t fc3_weight_size = fc3_weight_shape[0] * fc3_weight_shape[1];
        for (size_t i = 0; i < fc3_weight_size; ++i) {
            encoder.encode(trained_data[trained_data_index], SCALE, temp_plain);
            encryptor.encrypt_symmetric(temp_plain, temp_cipher);
            temp_cipher.save(fc3_weight_outstream);
            ++trained_data_index;
        }
        fc3_weight_outstream.close();

        // fc3_bias
        ofstream fc3_bias_outstream;
        string fc3_bias_path = dataset_path + files_prefix + string("_fc3_bias");
        if (work_mode == full_) {
            fc3_bias_path += string("_full");
        }
        fc3_bias_outstream.open(fc3_bias_path, ios::out | ios::binary);

        size_t fc3_bias_size = shape.fc_bias[3];
        for (size_t i = 0; i < fc3_bias_size; ++i) {
            encoder.encode(trained_data[trained_data_index], SCALE, temp_plain);
            encryptor.encrypt_symmetric(temp_plain, temp_cipher);
            temp_cipher.save(fc3_bias_outstream);
            ++trained_data_index;
        }
        fc3_bias_outstream.close();
    }
}

vector<double> read_conv_result(
    const string &dataset,
    SEALPACK &seal,
    array<size_t, 4> output_shape,
    size_t output_size,
    mode work_mode,
    int conv = 2) {

    const string result_path = DATA_PATH + string("communication/") + dataset +
        (conv == 1 ? string("_conv1_result") : string("_conv2_result"));

    ifstream result_stream;
    result_stream.open(result_path, ios::in | ios::binary);
    vector<double> output(output_size);
    vector<double> decrypt_results(output_size);

    auto encrypted_neurons = get_encrypted_neurons_list(dataset);
    const size_t batch_size = output_shape[0];
    size_t block_size = output_shape[2] * output_shape[3];
    for (size_t i = 0; i < output_size / batch_size;) {
        if (is_neuron_encrypted(encrypted_neurons, 2, i / block_size) && work_mode == separate_) {
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
}

vector<double> read_fc_result(SEALPACK &seal, array<size_t, 2> output_shape) {
    // MNIST only
    const string result_path = DATA_PATH + string("communication/MNIST_fc3_result");

    ifstream result_stream;
    result_stream.open(result_path, ios::in | ios::binary);
    vector<double> output(output_shape[0] * output_shape[1]);
    vector<double> decrypt_results(output_shape[0]);

    for (size_t i = 0; i < output_shape[1]; ++i) {
        seal.cipher_.load(seal.context_, result_stream);
        seal.decryptor_.decrypt(seal.cipher_, seal.plain_);
        seal.encoder_.decode(seal.plain_, decrypt_results);

        for (size_t j = 0; j < output_shape[0]; ++j) {
            output[i + j * output_shape[1]] = decrypt_results[j];
        }
    }

    result_stream.close();
    return output;
}

double *get_result(char *dataset, int batch_size, mode work_mode = separate_) {
    SEALPACK seal(work_mode);
    auto shapes = Shapes[string(dataset)];
    update_shape_size(shapes, batch_size);
    size_t output_size = batch_size *
        (work_mode == full_
             ? 10
             : shapes.conv_output[2][1] * shapes.conv_output[2][2] * shapes.conv_output[2][3]);

    vector<double> final_result;
    if (work_mode == separate_ || work_mode == remove_) {
        final_result =
            read_conv_result(dataset, seal, shapes.conv_output[2], output_size, work_mode);
    } else {
        final_result = read_fc_result(seal, array<size_t, 2>{(size_t)batch_size, 10});
    }

    if (dataset == string("TinyImageNet")) {
        size_t tiny_imagenet_middle_size = batch_size * shapes.conv_output[1][1] *
            shapes.conv_output[1][2] * shapes.conv_output[1][3];
        vector<double> middle_result =
            read_conv_result(dataset, seal, shapes.conv_output[2], output_size, work_mode, 1);

        double *output = new double[tiny_imagenet_middle_size + output_size];
        size_t i = 0;
        for (; i < tiny_imagenet_middle_size; ++i) {
            output[i] = middle_result[i];
        }

        for (size_t j = 0; j < output_size; ++i, ++j) {
            output[i] = final_result[j];
        }

        return output;
    } else {
        double *output = new double[output_size];
        for (size_t i = 0; i < output_size; ++i) {
            output[i] = final_result[i];
        }

        return output;
    }
}
}