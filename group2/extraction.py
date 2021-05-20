# utility functions
def get_rotation(signal_index):
    if signal_index == "1"
        return 45 degrees
    ....
def get_power(signal_index)
    if signal_index == "1"
        return 2.3


# main functions
def detection(samples, mod): # mod = "QAM64", ....
    # make sure it's normalized or has the correct size
    ideal_cnstln = get_constln(mod)
    ideal_cnstln = rotate(get_rotation(...), ideal_cnstln)

    # do nearest distance decoding
    return detected_samples


def map_to_bits(detected_samples):
    np.array([], dtype='uint8')
    return bitstream

# np.array with dtype='uint8'
def unscramble(bitstream):
    # e.g. bitstream = [1, 0, 1, 1, 1, ....]
    p = "1111111110000111101110000101100110110111101000011100110000100100010101110101111001001011100111000000111011101001111010100101000000101010101111101011010000011011101101101011000001011101111100011110011010011010111000110100010111111101001011000101001100011000000011001100101011001001111110110100100100110111111001011010100001010001001110110010111101100001101010100111001000011000100001000000001000100011001000111010101101100011100010010101000110110011111001111000101101110010100100000100110011101000111110111100000"
    unscrambled_bitstream = bistream xor p
    return unscrambled_bitstream

def map_to_image(bitstream):
    # e.g. bitstream = [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, ....]

    # convert to bytes
    # bytestream = [24, 255, 0, 3, 42, ...]

    # reshape to 256x256
    numpy_array.tofile("image.bin")

