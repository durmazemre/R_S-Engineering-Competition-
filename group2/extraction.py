import sys
import os
sys.path.append(os.path.abspath("../common/"))
import data_io_ingestion as io
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("../group1/"))
import get_samples



# utility functions
def get_rotation(signal_index):
    if signal_index == "1"
        return 45 degrees
    ....
def get_power(signal_index)
    if signal_index == "1"
        return 2.3


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

    return idx_list


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

    def sym2bitdict():
        # TODO: Find a way to append 0's
        input_data = io.get_set_two(cnstln=True)
        for k in range(0, 5):
            if k == 0:
                sep_bpsk_const = input_data[k]
                real_bpsk = sep_bpsk_const[0::2]
                imag_bpsk = sep_bpsk_const[1::2]
                bpsk_const = real_bpsk + 1j * imag_bpsk

                binary = np.array(np.zeros((2)))

                for j in range(2):
                    binary[j] = bin(int(j))[2:].zfill(8)

                bpsk_dict = dict(zip(binary, bpsk_const))

            elif k == 4:
                sep_qpsk_const = input_data[k]
                real_qpsk = sep_qpsk_const[0::2]
                imag_qpsk = sep_qpsk_const[1::2]
                qpsk_const = real_qpsk + 1j * imag_qpsk

                binary = np.array(np.zeros((4)))

                for j in range(4):
                    binary[j] = bin(int(j))[2:].zfill(8)

                qpsk_dict = dict(zip(binary, qpsk_const))
            elif k == 1:

                sep_16qam_const = input_data[k]
                real_16qam = sep_16qam_const[0::2]
                imag_16qam = sep_16qam_const[1::2]
                qam16_const = real_16qam + 1j * imag_16qam

                binary = np.array(np.zeros((16)))

                for j in range(16):
                    binary[j] = bin(int(j))[2:].zfill(8)

                qam16_dict = dict(zip(binary, qam16_const))

            elif k == 2:

                sep_32qam_const = input_data[k]
                real_32qam = sep_32qam_const[0::2]
                imag_32qam = sep_32qam_const[1::2]
                qam32_const = real_32qam + 1j * imag_32qam

                binary = np.array(np.zeros((32)))

                for j in range(32):
                    binary[j] = bin(int(j))[2:].zfill(8)

                qam32_dict = dict(zip(binary, qam32_const))

            elif k == 3:

                sep_64qam_const = input_data[k]
                real_64qam = sep_64qam_const[0::2]
                imag_64qam = sep_64qam_const[1::2]
                qam64_const = real_64qam + 1j * imag_64qam

                binary = np.array(np.zeros((64)))

                for j in range(64):
                    binary[j] = bin(int(j))[2:].zfill(8)

                qam64_dict = dict(zip(binary, qam64_const))

        return bpsk_dict, qpsk_dict, qam16_dict, qam32_dict, qam64_dict

