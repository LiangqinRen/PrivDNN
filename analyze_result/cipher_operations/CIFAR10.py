def count_cipher_addition(percent):
    cipher_operations = int(64 * percent / 100) * 32 * 32 * 9 * 3  # conv1
    cipher_operations += int(64 * percent / 100) * 32 * 32  # bn1
    cipher_operations += 0  # square
    cipher_operations += int(64 * percent / 100) * 16 * 16 * 9 * 64  # conv2
    cipher_operations += int(64 * percent / 100) * 16 * 16  # bn2
    cipher_operations += 0  # ReLU
    cipher_operations += 0  # pool

    total_operations = 64 * 32 * 32 * 9 * 3  # conv1
    total_operations += 64 * 32 * 32  # bn1
    total_operations += 0  # square
    total_operations += 64 * 32 * 32 * 9 * 64  # conv2
    total_operations += 64 * 32 * 32  # bn2
    total_operations += 0  # ReLU
    total_operations += 64 * 16 * 16 * 3  # pool2
    total_operations += 128 * 16 * 16 * 9 * 64  # conv3
    total_operations += 128 * 16 * 16  # bn3
    total_operations += 0  # ReLU
    total_operations += 128 * 16 * 16 * 9 * 128  # conv4
    total_operations += 128 * 16 * 16  # bn4
    total_operations += 0  # ReLU
    total_operations += 128 * 8 * 8 * 3  # pool4
    total_operations += 256 * 8 * 8 * 9 * 128  # conv5
    total_operations += 256 * 8 * 8  # bn5
    total_operations += 0  # ReLU
    total_operations += 256 * 8 * 8 * 9 * 256  # conv6
    total_operations += 256 * 8 * 8  # bn6
    total_operations += 0  # ReLU
    total_operations += 256 * 8 * 8 * 9 * 256  # conv7
    total_operations += 256 * 8 * 8  # bn7
    total_operations += 0  # ReLU
    total_operations += 256 * 4 * 4 * 3  # pool7
    total_operations += 512 * 4 * 4 * 9 * 256  # conv8
    total_operations += 512 * 4 * 4  # bn8
    total_operations += 0  # ReLU
    total_operations += 512 * 4 * 4 * 9 * 512  # conv9
    total_operations += 512 * 4 * 4  # bn9
    total_operations += 0  # ReLU
    total_operations += 512 * 4 * 4 * 9 * 512  # conv10
    total_operations += 512 * 4 * 4  # bn10
    total_operations += 0  # ReLU
    total_operations += 512 * 2 * 2 * 3  # pool10
    total_operations += 512 * 2 * 2 * 9 * 256  # conv11
    total_operations += 512 * 2 * 2  # bn11
    total_operations += 0  # ReLU
    total_operations += 512 * 2 * 2 * 9 * 512  # conv12
    total_operations += 512 * 2 * 2  # bn12
    total_operations += 0  # ReLU
    total_operations += 512 * 2 * 2 * 9 * 512  # conv13
    total_operations += 512 * 2 * 2  # bn13
    total_operations += 0  # ReLU
    total_operations += 512 * 1 * 1 * 3  # pool13
    total_operations += 1 * 511 * 512  # fc1
    total_operations += 1 * 511 * 512  # fc2
    total_operations += 1 * 511 * 10  # output

    return (cipher_operations, total_operations)


def count_cipher_multiplication(percent):
    cipher_operations = int(64 * percent / 100) * 32 * 32 * 9 * 3  # conv1
    cipher_operations += int(64 * percent / 100) * 32 * 32  # bn1
    cipher_operations += int(64 * percent / 100) * 32 * 32  # square
    cipher_operations += int(64 * percent / 100) * 16 * 16 * 9 * 64  # conv2
    cipher_operations += int(64 * percent / 100) * 16 * 16  # bn2
    cipher_operations += 0  # ReLU
    cipher_operations += 0  # pool

    total_operations = 64 * 32 * 32 * 9 * 3  # conv1
    total_operations += 64 * 32 * 32  # bn1
    total_operations += 64 * 32 * 32  # square
    total_operations += 64 * 32 * 32 * 9 * 64  # conv2
    total_operations += 64 * 32 * 32  # bn2
    total_operations += 0  # ReLU
    total_operations += 64 * 16 * 16 * 3  # pool2
    total_operations += 128 * 16 * 16 * 9 * 64  # conv3
    total_operations += 128 * 16 * 16  # bn3
    total_operations += 0  # ReLU
    total_operations += 128 * 16 * 16 * 9 * 128  # conv4
    total_operations += 128 * 16 * 16  # bn4
    total_operations += 0  # ReLU
    total_operations += 128 * 8 * 8 * 3  # pool4
    total_operations += 256 * 8 * 8 * 9 * 128  # conv5
    total_operations += 256 * 8 * 8  # bn5
    total_operations += 0  # ReLU
    total_operations += 256 * 8 * 8 * 9 * 256  # conv6
    total_operations += 256 * 8 * 8  # bn6
    total_operations += 0  # ReLU
    total_operations += 256 * 8 * 8 * 9 * 256  # conv7
    total_operations += 256 * 8 * 8  # bn7
    total_operations += 0  # ReLU
    total_operations += 256 * 4 * 4 * 3  # pool7
    total_operations += 512 * 4 * 4 * 9 * 256  # conv8
    total_operations += 512 * 4 * 4  # bn8
    total_operations += 0  # ReLU
    total_operations += 512 * 4 * 4 * 9 * 512  # conv9
    total_operations += 512 * 4 * 4  # bn9
    total_operations += 0  # ReLU
    total_operations += 512 * 4 * 4 * 9 * 512  # conv10
    total_operations += 512 * 4 * 4  # bn10
    total_operations += 0  # ReLU
    total_operations += 512 * 2 * 2 * 3  # pool10
    total_operations += 512 * 2 * 2 * 9 * 256  # conv11
    total_operations += 512 * 2 * 2  # bn11
    total_operations += 0  # ReLU
    total_operations += 512 * 2 * 2 * 9 * 512  # conv12
    total_operations += 512 * 2 * 2  # bn12
    total_operations += 0  # ReLU
    total_operations += 512 * 2 * 2 * 9 * 512  # conv13
    total_operations += 512 * 2 * 2  # bn13
    total_operations += 0  # ReLU
    total_operations += 512 * 1 * 1 * 3  # pool13
    total_operations += 1 * 512 * 512  # fc1
    total_operations += 1 * 512 * 512  # fc2
    total_operations += 1 * 512 * 10  # output

    return (cipher_operations, total_operations)


def count_neurons(percent):
    encrypted_neurons = int(64 * percent / 100) + int(64 * percent / 100)
    total_neurons = (
        64 + 64 + 128 + 128 + 256 + 256 + 256 + 512 + 512 + 512 + 512 + 512 + 512
    )

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
