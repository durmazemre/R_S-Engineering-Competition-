import sys
import os
sys.path.append(os.path.abspath("../common/"))
import data_io_ingestion as io
import numpy as np
import matplotlib.pyplot as plt
import math

sys.path.append(os.path.abspath("../group1/"))
import get_samples

M_to_data_index = {2: 0, 4: 4, 16:1, 32:2, 64:3} # dictionary: M -> input_data index array

def main_function():
    # INIT
    SIG_INDEX = 0
    I_results, Q_results = io.load_samples(0)
    I_results = I_results[:5]
    Q_results = Q_results[:5]
    # plt.scatter(I_results, Q_results)
    # plt.show()

    samples = I_results + 1j*Q_results
    samples = normalize_signal(samples)

    print(samples)
    M = get_cnstln_order(SIG_INDEX)
    k = M_to_data_index[M]
    input_data = io.get_set_two(cnstln=True)
    cnstln = input_data[k]
    cnstln_r = cnstln[0::2]
    cnstln_i = cnstln[1::2]
    cnstln = cnstln_r + 1j*cnstln_i
    cnstln = normalize_signal(cnstln)
    cnstln = rotate(cnstln, get_rotation(SIG_INDEX))
    print(cnstln)

    detected_samples = detection(samples, cnstln)
    print(detected_samples)

    # plt.scatter(np.real(cnstln), np.imag(cnstln))
    # plt.scatter(np.real(samples), np.imag(samples))
    # plt.scatter(np.real(detected_samples), np.imag(detected_samples))
    # plt.show()
    bitstream = map_to_bits(M, cnstln, detected_samples)
    print(bitstream)

    print("lasdjflkdsj")
    bitstream = unscramble(bitstream)
    print(bitstream)
    # TODO map to image 

# utility functions
def get_rotation(signal_index):
    if signal_index == 0:
        return 73.5 # degrees
# def get_power(signal_index)
#     if signal_index == "1":
#         return 2.3

def normalize_signal(samples):
    power = np.mean(np.abs(samples) ** 2)
    norm_samples = np.sqrt(1 / power) * samples
    return  norm_samples

def rotate(const,degree):
    rot_const = const * (np.cos(degree*np.pi/180) + 1j*np.sin(degree*np.pi/180))
    return rot_const

def get_cnstln_order(SIG_INDEX):
    if SIG_INDEX == 0:
        return 64
# main functions
def detection(samples, const):
    idx_list = np.array([])
    mod_ord = const.shape[0]

    for sample_idx in range(samples.shape[0]):

        diff_list = np.array([])
        for sym_idx in range(mod_ord):
            diff = np.absolute(const[sym_idx] - samples[sample_idx])
            diff_list = np.append(diff_list, diff)
        min_idx = np.argmin(diff_list)
        idx_list = np.append(idx_list, min_idx)
    # return idx_list
    idx_list = [int(x) for x in idx_list]
    return const[idx_list]


def map_to_bits(M, cnstln, detected_samples): # M is the mod order
    bitstream = np.array([], dtype='uint8')
    for s in detected_samples:
        # find index
        index = 0
        for i, cns in enumerate(cnstln):
            if cns == s:
                index = i
                break

        m = int(np.log2(M))
        bits = np.array(list(np.binary_repr(index).zfill(m))).astype(np.int8)
        bitstream = np.append(bitstream, bits)
    return bitstream


def unscramble(bitstream):
    # e.g. bitstream = [1, 0, 1, 1, 1, ....]
    p = "1111111110000111101110000101100110110111101000011100110000100100010101110101111001001011100111000000111011101001111010100101000000101010101111101011010000011011101101101011000001011101111100011110011010011010111000110100010111111101001011000101001100011000000011001100101011001001111110110100100100110111111001011010100001010001001110110010111101100001101010100111001000011000100001000000001000100011001000111010101101100011100010010101000110110011111001111000101101110010100100000100110011101000111110111100000"
    # unscrambled_bitstream = bool(bitstream) ^ bool(p)
    p = np.asarray(list(p))
    p = p.astype(np.int)
    LEN = len(p)
    print(p)
    size_fr = math.floor(len(bitstream)/len(p))
    out = []
    for i in range(size_fr):
        out.append(np.bitwise_xor(p, bitstream[i*LEN:(i+1)*LEN]))
    LEN = len(bitstream[i*LEN:])
    out.append(np.bitwise_xor(p[:LEN], bitstream[i*LEN:]))

    # n = 1 # n=0 possible but in our case not practical as an input for the bitstream
    # LEN = len(p)
    # for i in range(size_fr):
    #     out.append(np.bitwise_xor(p, bitstream[i*LEN:(i+1)*LEN]))

    return out[0]

def map_to_image(bitstream):
    # e.g. bitstream = [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, ....]

    # convert to bytes
    # bytestream = [24, 255, 0, 3, 42, ...]

    # reshape to 256x256
    numpy_array.tofile("image.bin")

if __name__ == '__main__':
    main_function()
