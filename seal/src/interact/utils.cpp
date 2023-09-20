#include "utils.h"

extern "C" {

void update_shape_size(Shape &shape, size_t batch_size) {
    shape.conv_input[1][0] = batch_size;
    shape.conv_input[2][0] = batch_size;

    shape.conv_output[1][0] = batch_size;
    shape.conv_output[2][0] = batch_size;

    shape.pool_input[1][0] = batch_size;
    shape.pool_input[2][0] = batch_size;

    shape.pool_output[1][0] = batch_size;
    shape.pool_output[2][0] = batch_size;
}

json get_encrypted_neurons_list(string dataset) {
    const static string NEURONS_LIST_PATH = string(PROJECT_PATH) + string("saved_models/") +
        dataset + ("/inference_encrypted_neurons.json");

    std::ifstream ifs(NEURONS_LIST_PATH);
    json encrypted_neurons_json;
    ifs >> encrypted_neurons_json;

    return encrypted_neurons_json;
}

bool is_neuron_encrypted(json encryped_neurons, size_t round, size_t neuron) {
    return count(
        encryped_neurons[to_string(round)].begin(),
        encryped_neurons[to_string(round)].end(),
        neuron);
}

bool is_file_exist(const string &file_name) {
    ifstream file_stream(file_name.c_str());
    return file_stream.good();
}

void save_parms(mode work_mode) {
    string PARMS_PATH = DATA_PATH + string("parms");
    if (work_mode == full_) {
        PARMS_PATH += string("_full");
    }

    if (!is_file_exist(PARMS_PATH)) {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = work_mode == full_ ? 32768 : 16384;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        if (work_mode == full_) {
            parms.set_coeff_modulus(CoeffModulus::Create(
                poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60}));
        } else {
            parms.set_coeff_modulus(
                CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 40, 40, 60}));
        }

        ofstream parms_stream;
        parms_stream.open(PARMS_PATH, ios::out | ios::binary);
        parms.save(parms_stream);
        parms_stream.close();
    }
}

EncryptionParameters read_parms(mode work_mode) {
    string PARMS_PATH = DATA_PATH + string("parms");
    if (work_mode == full_) {
        PARMS_PATH += string("_full");
    }

    EncryptionParameters parms;
    ifstream parms_stream;
    parms_stream.open(PARMS_PATH, ios::in | ios::binary);
    parms.load(parms_stream);
    parms_stream.close();

    return parms;
}

void save_keys(mode work_mode) {
    string secret_key_path = DATA_PATH + string("secret_key");
    if (work_mode == full_) {
        secret_key_path += string("_full");
    }
    if (!is_file_exist(secret_key_path)) {
        auto parms = read_parms(work_mode);
        SEALContext context(parms);
        KeyGenerator keygen(context);
        auto secret_key = keygen.secret_key();

        ofstream secret_key_stream;
        secret_key_stream.open(secret_key_path, ios::out | ios::binary);
        secret_key.save(secret_key_stream);
        secret_key_stream.close();
    }
}

SecretKey read_secret_key(mode work_mode) {
    string SECRET_KEY_PATH = DATA_PATH + string("secret_key");
    if (work_mode == full_) {
        SECRET_KEY_PATH += string("_full");
    }

    SecretKey secret_key;
    auto parms = read_parms(work_mode);
    SEALContext context(parms);

    ifstream secret_key_stream;
    secret_key_stream.open(SECRET_KEY_PATH, ios::in | ios::binary);
    secret_key.load(context, secret_key_stream);
    secret_key_stream.close();

    return secret_key;
}
}
