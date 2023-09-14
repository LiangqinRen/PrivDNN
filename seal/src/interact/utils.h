#include "common.h"
#include "configor/json.hpp"
#include "path.h"

#include <fstream>
#include <iostream>
#include <string>
#include <typeinfo>
#include <variant>

using namespace std;
using namespace configor;
using namespace seal;

extern "C" {
struct MNIST_Shape {
    unordered_map<size_t, size_t> kernal_sizes = {{0, 0}, {1, 5}, {2, 5}};
    unordered_map<size_t, array<size_t, 4>> conv_input = {{1, {1, 1, 28, 28}}, {2, {1, 6, 12, 12}}};
    unordered_map<size_t, array<size_t, 4>> conv_output = {{1, {1, 6, 24, 24}}, {2, {1, 16, 8, 8}}};
    unordered_map<size_t, array<size_t, 4>> conv_weight = {{1, {6, 1, 5, 5}}, {2, {16, 6, 5, 5}}};
    unordered_map<size_t, size_t> conv_bias = {{0, 0}, {1, 6}, {2, 16}};
    unordered_map<size_t, array<size_t, 4>> pool_input = {{1, {1, 6, 24, 24}}, {2, {1, 16, 8, 8}}};
    unordered_map<size_t, array<size_t, 4>> pool_output = {{1, {1, 6, 12, 12}}, {2, {1, 16, 4, 4}}};
    unordered_map<size_t, array<size_t, 2>> fc_weight = {
        {1, {256, 120}},
        {2, {120, 84}},
        {3, {84, 10}}};
    unordered_map<size_t, size_t> fc_bias = {{1, 120}, {2, 84}, {3, 10}};
};

const static string DATA_PATH = string(PROJECT_PATH) + string("/seal/data/");
const double SCALE = pow(2.0, 40);
enum mode { separate_, remove_, full_ };

json get_encrypted_neurons_list(string dataset);
bool is_neuron_encrypted(json encryped_neurons, size_t round, size_t neuron);

bool is_file_exist(const string &file_name);
void save_parms(mode work_mode);
void save_keys(mode work_mode);
EncryptionParameters read_parms(mode work_mode);
SecretKey read_secret_key(mode work_mode);

MNIST_Shape get_MNIST_shapes(int batch_size);

class SEALPACK {
public:
    SEALPACK(mode work_mode) {
        work_mode_ = work_mode;
        keygen_.create_relin_keys(relin_keys_);

        parms_ = read_parms(work_mode_);
        secret_key_ = read_secret_key(work_mode_);
    }

    EncryptionParameters parms_;
    SecretKey secret_key_;

    SEALContext context_ = SEALContext(parms_);
    CKKSEncoder encoder_ = CKKSEncoder(context_);
    Evaluator evaluator_ = Evaluator(context_);

    KeyGenerator keygen_ = KeyGenerator(context_, secret_key_);
    Encryptor encryptor_ = Encryptor(context_, secret_key_);
    Decryptor decryptor_ = Decryptor(context_, secret_key_);

    Plaintext plain_;
    Ciphertext cipher_;
    RelinKeys relin_keys_;

private:
    mode work_mode_;
};
}