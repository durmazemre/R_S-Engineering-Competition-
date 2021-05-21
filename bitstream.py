import numpy as np
import math

# np.array with dtype='uint8'
def unscramble(bitstream):
    # e.g. bitstream = [1, 0, 1, 1, 1, ....]
    p = "1111111110000111101110000101100110110111101000011100110000100100010101110101111001001011100111000000111011101001111010100101000000101010101111101011010000011011101101101011000001011101111100011110011010011010111000110100010111111101001011000101001100011000000011001100101011001001111110110100100100110111111001011010100001010001001110110010111101100001101010100111001000011000100001000000001000100011001000111010101101100011100010010101000110110011111001111000101101110010100100000100110011101000111110111100000"
    # unscrambled_bitstream = bool(bitstream) ^ bool(p)
    p = np.asarray(list(p))
    p = p.astype(np.int)
    size_fr = math.floor(len(bitstream)/len(p))
    out = []
    n = 1 # n=0 possible but in our case not practical as an input for the bitstream
    while n <= size_fr:
        k = len(p)
        for i in p:
            out.append(np.bitwise_xor(i, bitstream[i]))
        n += 1
        bitstream = bitstream[k:]
        n = n + 1    
    return out


############################################
## test start
############################################
bitstream = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,0,1,1,0,1,0,1,1,0,1,1,0,1,0,1,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1])
print(len(bitstream))
out = unscramble(bitstream)
## test over
############################################

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

bytes_stream = map_to_image(out)

out_new = map_to_image(out)
print(type(out_new))
print(bytes_stream)
print(len(out))
