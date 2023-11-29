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

void print_current_time();

struct Shape {
    unordered_map<size_t, size_t> kernal_sizes;
    unordered_map<size_t, array<size_t, 4>> conv_input;
    unordered_map<size_t, array<size_t, 4>> conv_output;
    unordered_map<size_t, array<size_t, 4>> conv_weight;
    unordered_map<size_t, size_t> conv_bias;
    unordered_map<size_t, array<size_t, 4>> pool_input;
    unordered_map<size_t, array<size_t, 4>> pool_output;
    unordered_map<size_t, array<size_t, 2>> fc_weight;
    unordered_map<size_t, size_t> fc_bias;
};

const Shape mnist{
    {{1, 5}, {2, 5}},
    {{1, {1, 1, 28, 28}}, {2, {1, 6, 12, 12}}},
    {{1, {1, 6, 24, 24}}, {2, {1, 16, 8, 8}}},
    {{1, {6, 1, 5, 5}}, {2, {16, 6, 5, 5}}},
    {{1, 6}, {2, 16}},
    {{1, {1, 6, 24, 24}}, {2, {1, 16, 8, 8}}},
    {{1, {1, 6, 12, 12}}, {2, {1, 16, 4, 4}}},
    {{1, {256, 120}}, {2, {120, 84}}, {3, {84, 10}}},
    {{1, 120}, {2, 84}, {3, 10}}};

const Shape emnist{
    {{1, 5}, {2, 5}},
    {{1, {1, 1, 28, 28}}, {2, {1, 10, 12, 12}}},
    {{1, {1, 10, 24, 24}}, {2, {1, 20, 8, 8}}},
    {{1, {10, 1, 5, 5}}, {2, {20, 10, 5, 5}}},
    {{1, 10}, {2, 20}},
    {{1, {1, 10, 24, 24}}, {2, {1, 20, 8, 8}}},
    {{1, {1, 96, 15, 15}}, {2, {1, 256, 7, 7}}},
};

const Shape gtsrb{
    {{1, 5}, {2, 3}},
    {{1, {1, 3, 32, 32}}, {2, {1, 96, 15, 15}}},
    {{1, {1, 96, 30, 30}}, {2, {1, 256, 15, 15}}},
    {{1, {96, 3, 5, 5}}, {2, {256, 96, 3, 3}}},
    {{1, 96}, {2, 256}},
    {{1, {1, 96, 30, 30}}, {2, {1, 256, 15, 15}}},
    {{1, {1, 96, 15, 15}}, {2, {1, 256, 7, 7}}},
};

const Shape cifar10{
    {{1, 3}, {2, 3}},
    {{1, {1, 3, 32, 32}}, {2, {1, 64, 32, 32}}},
    {{1, {1, 64, 32, 32}}, {2, {1, 64, 32, 32}}},
    {{1, {64, 3, 3, 3}}, {2, {64, 64, 3, 3}}},
    {{1, 64}, {2, 64}},
};

static unordered_map<string, struct Shape> Shapes{
    {string("MNIST"), mnist},
    {string("EMNIST"), emnist},
    {string("GTSRB"), gtsrb},
    {string("CIFAR10"), cifar10}};

void update_shape_size(Shape &shape, size_t batch_size);

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

class SEALPACK {
public:
    SEALPACK(mode work_mode) : work_mode_(work_mode) {
        keygen_.create_relin_keys(relin_keys_);
    }
    mode work_mode_;
    EncryptionParameters parms_ = read_parms(work_mode_);
    SecretKey secret_key_ = read_secret_key(work_mode_);

    SEALContext context_ = SEALContext(parms_);
    CKKSEncoder encoder_ = CKKSEncoder(context_);
    Evaluator evaluator_ = Evaluator(context_);

    KeyGenerator keygen_ = KeyGenerator(context_, secret_key_);
    Encryptor encryptor_ = Encryptor(context_, secret_key_);
    Decryptor decryptor_ = Decryptor(context_, secret_key_);

    Plaintext plain_;
    Ciphertext cipher_;
    RelinKeys relin_keys_;
};
}
