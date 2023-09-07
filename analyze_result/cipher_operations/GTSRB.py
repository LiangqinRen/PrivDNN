print("cipher addition")
print(
    int(96 * 0.5) * 26 * 26 * 25  # conv1
    + int(96 * 0.5) * 13 * 13 * 3  # pool
    + 0  # square
    + int(256 * 0.5) * 13 * 13 * 9 * 96  # conv2
    + 0  # pool
    + 0  # relu
)


print("total addition")
print(
    96 * 26 * 26 * 25  # conv1
    + 96 * 13 * 13 * 3  # pool
    + 0  # square
    + 256 * 13 * 13 * 9 * 96  # conv2
    + 0  # pool
    + 0  # relu
    + 384 * 6 * 6 * 9 * 256  # conv3
    + 0  # relu
    + 384 * 6 * 6 * 9 * 256  # conv4
    + 0  # relu
    + 256 * 6 * 6 * 9 * 384  # conv5
    + 0  # relu
    + 1 * 2303 * 512  # fc1
    + 1 * 511 * 128  # fc2
    + 1 * 127 * 43  # output
)


print("cipher multiplication")
print(
    int(96 * 0.5) * 26 * 26 * 25  # conv1
    + int(96 * 0.5) * 26 * 26  # square
    + int(96 * 0.5) * 13 * 13  # pool
    + int(256 * 0.5) * 13 * 13 * 9 * 96  # conv2
)


print("total multiplication")
print(
    96 * 26 * 26 * 25  # conv1
    + 96 * 26 * 26  # square
    + 96 * 13 * 13  # pool
    + 256 * 13 * 13 * 9 * 96  # conv2
    + 0  # relu
    + 384 * 6 * 6 * 9 * 256  # conv3
    + 0  # relu
    + 384 * 6 * 6 * 9 * 256  # conv4
    + 0  # relu
    + 256 * 6 * 6 * 9 * 384  # conv5
    + 0  # relu
    + 1 * 2304 * 512  # fc1
    + 1 * 512 * 128  # fc2
    + 1 * 128 * 43  # output
)
