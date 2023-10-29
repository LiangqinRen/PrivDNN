def count_cipher_addition(n11, n21):
    cipher_operations = n11 * 24 * 24 * 25  # conv1
    cipher_operations += n11 * 12 * 12 * 3  # pool
    cipher_operations += 0  # square
    cipher_operations += n21 * 8 * 8 * 25 * 6  # conv2
    cipher_operations += n21 * 4 * 4 * 3  # pool
    cipher_operations += 0  # square

    total_operations = 10 * 24 * 24 * 25  # conv1
    total_operations += 10 * 12 * 12 * 3  # pool
    total_operations += 0  # square
    total_operations += 20 * 8 * 8 * 25 * 6  # conv2
    total_operations += 20 * 4 * 4 * 3  # pool
    total_operations += 0  # square
    total_operations += 1 * 319 * 120  # fc1
    total_operations += 1 * 119 * 84  # fc2
    total_operations += 1 * 83 * 26  # output

    return (cipher_operations, total_operations)


def count_cipher_multiplication(n11, n21):
    cipher_operations = n11 * 24 * 24 * 25  # conv1
    cipher_operations += n11 * 12 * 12  # pool
    cipher_operations += n11 * 12 * 12  # square
    cipher_operations += n21 * 8 * 8 * 25 * 6  # conv2
    cipher_operations += n21 * 4 * 4  # pool
    cipher_operations += n21 * 4 * 4  # square

    total_operations = 10 * 24 * 24 * 25  # conv1
    total_operations += 10 * 12 * 12  # pool
    total_operations += 10 * 12 * 12  # square
    total_operations += 20 * 8 * 8 * 25 * 6  # conv2
    total_operations += 20 * 4 * 4  # pool
    total_operations += 20 * 4 * 4  # square
    total_operations += 1 * 320 * 120  # fc1
    total_operations += 1 * 120 * 84  # fc2
    total_operations += 1 * 84 * 26  # output

    return (cipher_operations, total_operations)


if __name__ == "__main__":
    for i in range(1, 11):
        for j in range(1, 21):
            (cipher_operations, total_operations) = count_cipher_addition(i, j)
            print(
                f"{i},{j:2} + {cipher_operations/total_operations*100:.5f}% {cipher_operations}/{total_operations}"
            )
            (cipher_operations, total_operations) = count_cipher_multiplication(i, j)
            print(
                f"{i},{j:2} * {cipher_operations/total_operations*100:.5f}% {cipher_operations}/{total_operations}"
            )

            print("-" * 31)
