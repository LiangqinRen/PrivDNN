def count_cipher_addition(percent):
    cipher_operations = int(96 * percent / 100) * 26 * 26 * 25  # conv1
    cipher_operations += int(96 * percent / 100) * 13 * 13 * 3  # pool
    cipher_operations += 0  # square
    cipher_operations += int(256 * percent / 100) * 13 * 13 * 9 * 96  # conv2
    cipher_operations += 0  # pool
    cipher_operations += 0  # ReLU

    total_operations = 96 * 26 * 26 * 25  # conv1
    total_operations += 96 * 13 * 13 * 3  # pool
    total_operations += 0  # square
    total_operations += 256 * 13 * 13 * 9 * 96  # conv2
    total_operations += 0  # pool
    total_operations += 0  # ReLU
    total_operations += 384 * 6 * 6 * 9 * 256  # conv3
    total_operations += 0  # ReLU
    total_operations += 384 * 6 * 6 * 9 * 256  # conv4
    total_operations += 0  # ReLU
    total_operations += 256 * 6 * 6 * 9 * 384  # conv5
    total_operations += 0  # ReLU
    total_operations += 1 * 2303 * 512  # fc1
    total_operations += 1 * 511 * 128  # fc2
    total_operations += 1 * 127 * 43  # output

    return (cipher_operations, total_operations)


def count_cipher_multiplication(percent):
    cipher_operations = int(96 * percent / 100) * 26 * 26 * 25  # conv1
    cipher_operations += int(96 * percent / 100) * 26 * 26  # square
    cipher_operations += int(96 * percent / 100) * 13 * 13  # pool
    cipher_operations += int(256 * percent / 100) * 13 * 13 * 9 * 96  # conv2

    total_operations = 96 * 26 * 26 * 25  # conv1
    total_operations += 96 * 26 * 26  # square
    total_operations += 96 * 13 * 13  # pool
    total_operations += 256 * 13 * 13 * 9 * 96  # conv2
    total_operations += 0  # ReLU
    total_operations += 384 * 6 * 6 * 9 * 256  # conv3
    total_operations += 0  # ReLU
    total_operations += 384 * 6 * 6 * 9 * 256  # conv4
    total_operations += 0  # ReLU
    total_operations += 256 * 6 * 6 * 9 * 384  # conv5
    total_operations += 0  # ReLU
    total_operations += 1 * 2304 * 512  # fc1
    total_operations += 1 * 512 * 128  # fc2
    total_operations += 1 * 128 * 43  # output

    return (cipher_operations, total_operations)


def count_neurons(percent):
    encrypted_neurons = int(96 * percent / 100) + int(256 * percent / 100)
    total_neurons = 96 + 256 + 384 + 384 + 256

    return (encrypted_neurons, total_neurons)


if __name__ == "__main__":
    for i in range(50, 105, 5):
        (encrypted_neurons, total_neurons) = count_neurons(i)
        print(
            f"{i:3}% n {encrypted_neurons/total_neurons*100:.3f}% {encrypted_neurons}/{total_neurons}"
        )

        (cipher_operations, total_operations) = count_cipher_addition(i)
        print(
            f"{i:3}% + {cipher_operations/total_operations*100:.5f}% {cipher_operations}/{total_operations}"
        )

        (cipher_operations, total_operations) = count_cipher_multiplication(i)
        print(
            f"{i:3}% * {cipher_operations/total_operations*100:.5f}% {cipher_operations}/{total_operations}"
        )

        print("-" * 34)
