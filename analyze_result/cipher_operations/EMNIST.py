print("cipher addition")
print(
    1 * 24 * 24 * 25  # conv1
    + 1 * 12 * 12 * 3  # pool
    + 0  # square
    + 3 * 8 * 8 * 25 * 6  # conv2
    + 3 * 4 * 4 * 3  # pool
    + 0  # square
)


print("total addition")
print(
    10 * 24 * 24 * 25  # conv1
    + 10 * 12 * 12 * 3  # pool
    + 0  # square
    + 20 * 8 * 8 * 25 * 6  # conv2
    + 20 * 4 * 4 * 3  # pool
    + 0  # square
    + 1 * 319 * 120  # fc1
    + 1 * 119 * 84  # fc2
    + 1 * 83 * 26  # output
)


print("cipher multiplication")
print(
    1 * 24 * 24 * 25  # conv1
    + 1 * 12 * 12  # pool
    + 1 * 12 * 12  # square
    + 3 * 8 * 8 * 25 * 6  # conv2
    + 3 * 4 * 4  # pool
    + 3 * 4 * 4  # square
)

print("total multiplication")
print(
    10 * 24 * 24 * 25  # conv1
    + 10 * 12 * 12  # pool
    + 10 * 12 * 12  # square
    + 20 * 8 * 8 * 25 * 6  # conv2
    + 20 * 4 * 4  # pool
    + 20 * 4 * 4  # square
    + 1 * 320 * 120  # fc1
    + 1 * 120 * 84  # fc2
    + 1 * 84 * 26  # output
)
