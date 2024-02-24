def count_cipher_addition(percent):
    cipher_operations = int(64 * percent / 100) * 64 * 64 * 9 * 3  # conv entry
    cipher_operations += int(64 * percent / 100) * 64 * 64  # bn entry
    cipher_operations += 0  # square
    cipher_operations += int(64 * percent / 100) * 64 * 64 * 9 * 64  # conv1
    cipher_operations += int(64 * percent / 100) * 64 * 64  # bn1
    cipher_operations += 0  # ReLU

    # entry
    total_operations = 64 * 64 * 64 * 9 * 3  # conv entry
    total_operations += 64 * 64 * 64  # bn entry
    total_operations += 0  # square
    total_operations += 64 * 64 * 64 * 9 * 64  # conv1
    total_operations += 64 * 64 * 64  # bn entry

    # layer 1
    total_operations += 64 * 64 * 64 * 9 * 64  # conv1
    total_operations += 64 * 64 * 64  # bn1
    total_operations += 0  # ReLU
    total_operations += 64 * 64 * 64 * 9 * 64  # conv2
    total_operations += 64 * 64 * 64  # bn2
    total_operations += 64 * 64 * 64 * 1 * 64  # shortcut

    # layer 2
    total_operations += 128 * 32 * 32 * 9 * 64  # conv1
    total_operations += 128 * 32 * 32  # bn1
    total_operations += 0  # ReLU
    total_operations += 128 * 32 * 32 * 9 * 128  # conv2
    total_operations += 128 * 32 * 32  # bn2
    total_operations += 128 * 32 * 32 * 1 * 128  # shortcut
    total_operations += 128 * 32 * 32  # bn
    total_operations += 128 * 32 * 32 * 9 * 128  # conv1
    total_operations += 128 * 32 * 32  # bn1
    total_operations += 0  # ReLU
    total_operations += 128 * 32 * 32 * 9 * 128  # conv2
    total_operations += 128 * 32 * 32  # bn2

    # layer 3
    total_operations += 256 * 16 * 16 * 9 * 128  # conv1
    total_operations += 256 * 16 * 16  # bn1
    total_operations += 0  # ReLU
    total_operations += 256 * 16 * 16 * 9 * 256  # conv2
    total_operations += 256 * 16 * 16  # bn2
    total_operations += 256 * 16 * 16 * 1 * 256  # shortcut
    total_operations += 256 * 16 * 16  # bn
    total_operations += 256 * 16 * 16 * 9 * 256  # conv1
    total_operations += 256 * 16 * 16  # bn1
    total_operations += 0  # ReLU
    total_operations += 256 * 16 * 16 * 9 * 256  # conv2
    total_operations += 256 * 16 * 16  # bn2

    # layer 4
    total_operations += 512 * 8 * 8 * 9 * 256  # conv1
    total_operations += 512 * 8 * 8  # bn1
    total_operations += 0  # ReLU
    total_operations += 512 * 8 * 8 * 9 * 512  # conv2
    total_operations += 512 * 8 * 8  # bn2
    total_operations += 512 * 8 * 8 * 1 * 512  # shortcut
    total_operations += 512 * 8 * 8  # bn
    total_operations += 512 * 8 * 8 * 9 * 256  # conv1
    total_operations += 512 * 8 * 8  # bn1
    total_operations += 0  # ReLU
    total_operations += 512 * 8 * 8 * 9 * 512  # conv2
    total_operations += 512 * 8 * 8  # bn2

    return (cipher_operations, total_operations)


def count_cipher_multiplication(percent):
    cipher_operations = int(64 * percent / 100) * 64 * 64 * 9 * 3  # conv entry
    cipher_operations += int(64 * percent / 100) * 64 * 64  # bn entry
    cipher_operations += 64 * 64  # square
    cipher_operations += int(64 * percent / 100) * 64 * 64 * 9 * 64  # conv1
    cipher_operations += int(64 * percent / 100) * 64 * 64  # bn1
    cipher_operations += 0  # ReLU

    # entry
    total_operations = 64 * 64 * 64 * 9 * 3  # conv entry
    total_operations += 64 * 64 * 64  # bn entry
    total_operations += 0  # square
    total_operations += 64 * 64 * 64 * 9 * 64  # conv1
    total_operations += 64 * 64 * 64  # bn entry

    # layer 1
    total_operations += 64 * 64 * 64 * 9 * 64  # conv1
    total_operations += 64 * 64 * 64  # bn1
    total_operations += 0  # ReLU
    total_operations += 64 * 64 * 64 * 9 * 64  # conv2
    total_operations += 64 * 64 * 64  # bn2
    total_operations += 64 * 64 * 64 * 1 * 64  # shortcut

    # layer 2
    total_operations += 128 * 32 * 32 * 9 * 64  # conv1
    total_operations += 128 * 32 * 32  # bn1
    total_operations += 0  # ReLU
    total_operations += 128 * 32 * 32 * 9 * 128  # conv2
    total_operations += 128 * 32 * 32  # bn2
    total_operations += 128 * 32 * 32 * 1 * 128  # shortcut
    total_operations += 128 * 32 * 32  # bn
    total_operations += 128 * 32 * 32 * 9 * 128  # conv1
    total_operations += 128 * 32 * 32  # bn1
    total_operations += 0  # ReLU
    total_operations += 128 * 32 * 32 * 9 * 128  # conv2
    total_operations += 128 * 32 * 32  # bn2

    # layer 3
    total_operations += 256 * 16 * 16 * 9 * 128  # conv1
    total_operations += 256 * 16 * 16  # bn1
    total_operations += 0  # ReLU
    total_operations += 256 * 16 * 16 * 9 * 256  # conv2
    total_operations += 256 * 16 * 16  # bn2
    total_operations += 256 * 16 * 16 * 1 * 256  # shortcut
    total_operations += 256 * 16 * 16  # bn
    total_operations += 256 * 16 * 16 * 9 * 256  # conv1
    total_operations += 256 * 16 * 16  # bn1
    total_operations += 0  # ReLU
    total_operations += 256 * 16 * 16 * 9 * 256  # conv2
    total_operations += 256 * 16 * 16  # bn2

    # layer 4
    total_operations += 512 * 8 * 8 * 9 * 256  # conv1
    total_operations += 512 * 8 * 8  # bn1
    total_operations += 0  # ReLU
    total_operations += 512 * 8 * 8 * 9 * 512  # conv2
    total_operations += 512 * 8 * 8  # bn2
    total_operations += 512 * 8 * 8 * 1 * 512  # shortcut
    total_operations += 512 * 8 * 8  # bn
    total_operations += 512 * 8 * 8 * 9 * 256  # conv1
    total_operations += 512 * 8 * 8  # bn1
    total_operations += 0  # ReLU
    total_operations += 512 * 8 * 8 * 9 * 512  # conv2
    total_operations += 512 * 8 * 8  # bn2

    # fc
    total_operations += 1 * 2048 * 200

    return (cipher_operations, total_operations)


def count_neurons(percent):
    encrypted_neurons = int(64 * percent / 100) + int(64 * percent / 100)
    total_neurons = 64 + 64 * 2 + 128 * 4 + 256 * 4 + 512 * 4

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
