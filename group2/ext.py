import sys
import os
sys.path.append(os.path.abspath("../common/"))
import data_io_ingestion as io
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("../group1/"))
import get_samples


def sym2bitdict():
    #TODO: Find a way to append 0's
    input_data = io.get_set_two(cnstln=True)
    for k in range(0, 5):
        if k == 0:
            sep_bpsk_const = input_data[k]
            real_bpsk = sep_bpsk_const[0::2]
            imag_bpsk = sep_bpsk_const[1::2]
            bpsk_const = real_bpsk + 1j * imag_bpsk

            binary = np.array(np.zeros((2)))

            for j in range(2):
                binary[j] = np.array(list(np.binary_repr(j).zfill(1))).astype(np.int8)
                # TODO: np.array(list(np.binary_repr(j).zfill(1))).astype(np.int8)

            bpsk_dict = dict(zip(binary, bpsk_const))

        elif k == 4:
            sep_qpsk_const = input_data[k]
            real_qpsk = sep_qpsk_const[0::2]
            imag_qpsk = sep_qpsk_const[1::2]
            qpsk_const = real_qpsk + 1j * imag_qpsk

            binary = np.array(np.zeros((4)))

            for j in range(4):
                binary[j] = bin(int(j))[2:].zfill(8)
                #np.array(list(np.binary_repr(j).zfill(2))).astype(np.int8)
                #np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)

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

def rotation(const,degree):
    rot_const = const * (np.cos(degree, dtype='d') + 1j * np.sin(degree, dtype='d'))
    return rot_const

def normalize_signal(samples):
    power = np.mean(np.abs(samples) ** 2)
    norm_samples = np.sqrt(1 / power) * samples
    return  norm_samples

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

if __name__ == '__main__':

    bpsk_dic, qpsk_dic, qam16_dic,qam32_dic, qam64_dic = sym2bitdict()


    input_data = io.get_set_two()  # 5 signals
    SIG_INDEX = 0
    sig = input_data[SIG_INDEX]  # signal

    # SAMPLING
    length = int(min(len(sig), 10e3) // 2)  # max value is 200000 i.e 400000//2
    throw_out_fraction = 0.3  # should be between 0 and 1
    I_results, Q_results = get_samples.get_samples(sig, length, throw_out_fraction)
    #plt.scatter(I_results, Q_results, color = 'g')
    #plt.ioff()
    #plt.show()

    samples = I_results + 1j * Q_results

    if SIG_INDEX == 0:

        mod_ord = 64

        myList = qam64_dic.items()
        bits, const = zip(*myList)

        norm_samples = normalize_signal(samples)


        const = np.asarray(const)

        norm_const = normalize_signal(const)

        rot_norm_samples = rotation(norm_samples,5)

        rot_norm_samples = normalize_signal(rot_norm_samples)

        plt.scatter(np.real(rot_norm_samples), np.imag(rot_norm_samples), color='b')
        #plt.scatter(np.real(norm_samples), np.imag(norm_samples), color='g')
        plt.scatter(np.real(norm_const), np.imag(norm_const),color='r')
        plt.show()

        print(samples.shape[0])

        detection(rot_norm_samples, norm_const)






    """
0            40             64QAM
1            40             16QAM
2            31.25          32QAM (probably)
3            12.5           64QAM
4            25             16QAM """

