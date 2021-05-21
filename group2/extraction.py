import sys
import os
sys.path.append(os.path.abspath("../common/"))
import data_io_ingestion as io
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("../group1/"))
import get_samples
from myutilities import largest_indices

SYNC = "010001000100001011101011111111001110100000111110110101011000000001101111010111010101001011111111111001111111101011010110"
M_to_data_index = {2: 0, 4: 4, 16:1, 32:2, 64:3} # dictionary: M -> input_data index array


def main_function():
    # INIT
    SIG_INDEX = 1
    I_results, Q_results = io.load_samples(SIG_INDEX)
    print("total number of samples", len(I_results))
    length = len(I_results)
    I_results = I_results[:length]
    Q_results = Q_results[:length]
    # plt.scatter(I_results, Q_results)
    # plt.show()

    samples = I_results + 1j*Q_results
    samples = samples/np.sqrt(get_known_power(SIG_INDEX))

    # print(samples)
    M = get_cnstln_order(SIG_INDEX)
    k = M_to_data_index[M]
    input_data = io.get_set_two(cnstln=True)
    cnstln = input_data[k]
    cnstln_r = cnstln[0::2]
    cnstln_i = cnstln[1::2]
    cnstln = cnstln_r + 1j*cnstln_i
    cnstln = normalize_signal(cnstln)
    rotation = get_known_rotation(SIG_INDEX)
    print("rotation", rotation)
    cnstln = rotate(cnstln, rotation)
    # print(cnstln)

    detected_samples = detection(samples, cnstln)
    # print(detected_samples)

    # plt.scatter(np.real(samples), np.imag(samples))
    # plt.scatter(np.real(detected_samples), np.imag(detected_samples))
    # plt.scatter(np.real(cnstln), np.imag(cnstln), marker='x')
    # plt.show()

    bitstream = map_to_bits(M, cnstln, detected_samples)
    # print(bitstream)
    sync_index = get_sync_index(bitstream, SIG_INDEX)

    index = sync_index + 120
    LEN_PAY = 65536*8
    ZERO_PAD = 0 # TODO
    print(len(bitstream))
    payload = bitstream[index + ZERO_PAD:index + ZERO_PAD + LEN_PAY]
    print(len(payload))
    unsc_payload = unscramble(payload)

    # tpay = np.array([1,0,1,0,1,1,1,0])
    # tpay2 = np.array([1,0,1,0,1,1,0,1])
    # tpay3 = np.array([1,1,1,1,1,1,1,1])
    # tpay3 = np.array([0,0,0,0,0,0,1,1])
    # tpay = np.append(tpay, tpay2)
    # tpay = np.append(tpay, tpay3)
    # print(tpay)
    save_image(unsc_payload)

    # TODO endianness?
    # print(bitstream)
    # TODO map to image 

def save_image(payload):
    assert len(payload) % 8 == 0
    payload = np.reshape(payload, (-1, 8))
    payload = np.packbits(payload, axis=-1)

    bytes_stream = [x[0] for x in payload]
    image = np.reshape(bytes_stream,(256,256))
    image.tofile("image" + str(SIG_INDEX) + ".bin")

def get_sync_index(bitstream, SIG_INDEX):
    if SIG_INDEX == 0: # for sig0.npy
        return 30738
    if SIG_INDEX == 1: # for sig1.npy
        return 128428 # 652836 (not 1177244 because not enough samples after it) # TODO try others?

    unitstream = (-1)**bitstream
    print("PAYLOAD FUNCION: ", "length of bitstream", len(bitstream))
    sync = np.fromstring(SYNC,'u1') - ord('0')
    unitsync = (-1)**sync
    corr = np.correlate(unitstream, unitsync)
    print("PAYLOAD FUNCION: ", "length of corr output", len(corr))
    # plt.plot(range(len(corr)), corr)
    # plt.show()
    idx = largest_indices(corr, 3)[0]
    print(idx)
    print(corr[idx])
    print("corr surroundings")
    for i in idx:
        # print(corr[i:i+140])
        print(unitstream[i:i+120])
        print("AFTER")
        print(unitstream[i+120:i+120+20])
        print(unitsync.dot(unitstream[i:i+120]).sum())
    return idx[0]



def get_known_rotation(SIG_INDEX):
    if SIG_INDEX == 0:
        return 73.5 + 180 # degrees
    elif SIG_INDEX == 1:
        return 180 + 65 # degrees
        # indices: [ 128428 1177244  652836]
        # correlation: [114 114 114]

def get_known_power(SIG_INDEX):
    if SIG_INDEX == 0:
        return 0.002493394838359623
    if SIG_INDEX == 1:
        return 0.002648637327440266

def get_power(samples):
    return np.mean(np.abs(samples) ** 2)

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
    elif SIG_INDEX == 1:
        return 16
    elif SIG_INDEX == 2:
        return 32 # we think

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
    LEN_p = len(p)
    size_fr = len(bitstream)//len(p)
    out = np.array([], dtype='uint8')
    for i in range(size_fr):
        out = np.append(out, np.bitwise_xor(p, bitstream[i*LEN_p:(i+1)*LEN_p]))
    LEN = len(bitstream[size_fr*LEN_p:])
    out = np.append(out, np.bitwise_xor(p[:LEN], bitstream[size_fr*LEN_p:]))

    return out


def map_to_image(out):
    bytes_stream = []
    bin_byte = []
    for i in out:
        i = str(i)
        bin_byte.append(i)
        if len(bin_byte) == 8:
            bin_con = int(bin_byte, 2)
            length = math.ceil(math.log(bin_con, 256))
            bin_con = int.to_bytes(bin_con, length=length, byteorder='big', signed=False)
            
            bin_byte = []

    # image = numpy.reshape(bytes_stream,(256,256))
    # image.tofile("image.bin")

    return bytes_stream

# def map_to_image(bitstream):
    # e.g. bitstream = [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, ....]

    # convert to bytes
    # bytestream = [24, 255, 0, 3, 42, ...]

    # reshape to 256x256
    # numpy_array.tofile("image.bin")

if __name__ == '__main__':
    main_function()
