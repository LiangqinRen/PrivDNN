print("cipher addition")
print(
    1 * 24 * 24 * 25  # conv1
    + 1 * 12 * 12 * 3  # pool
    + 0  # square
    + 5 * 8 * 8 * 25 * 6  # conv2
    + 5 * 4 * 4 * 3  # pool
    + 0  # square
)


print("total addition")
print(
    6 * 24 * 24 * 25  # conv1
    + 6 * 12 * 12 * 3  # pool
    + 0  # square
    + 16 * 8 * 8 * 25 * 6  # conv2
    + 16 * 4 * 4 * 3  # pool
    + 0  # square
    + 1 * 255 * 120  # fc1
    + 1 * 119 * 84  # fc2
    + 1 * 83 * 10  # output
)


print("cipher multiplication")
print(
    1 * 24 * 24 * 25  # conv1
    + 1 * 12 * 12  # pool
    + 1 * 12 * 12  # square
    + 5 * 8 * 8 * 25 * 6  # conv2
    + 5 * 4 * 4  # pool
    + 5 * 4 * 4  # square
)

print("total multiplication")
print(
    6 * 24 * 24 * 25  # conv1
    + 6 * 12 * 12  # pool
    + 6 * 12 * 12  # square
    + 16 * 8 * 8 * 25 * 6  # conv2
    + 16 * 4 * 4  # pool
    + 16 * 4 * 4  # square
    + 1 * 256 * 120  # fc1
    + 1 * 120 * 84  # fc2
    + 1 * 84 * 10  # output
)
