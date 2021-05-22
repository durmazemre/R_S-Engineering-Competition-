"""
Purpose: Machine learning (neural network) implementation to do constellation recognition.
Warning: Implementation is incomplete.
"""
from operator import index
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans
import sys
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

def channel(SNR_dB, sig_pwr, ch_in, rotation):
    n = ch_in.shape[1]
    if SNR_dB is not None:

        SNR = 10 ** (SNR_dB / 10)
        noise_pwr = sig_pwr / SNR

        noise = np.sqrt(noise_pwr / 2) * (np.random.randn(1, n) + 1j * np.random.randn(1, n))

    else:
        noise = np.zeros((1, n))

    if None != rotation:
        ch_out = ch_in * (np.cos(rotation, dtype='d') + 1j * np.sin(rotation, dtype='d'))
    else:
        ch_out = ch_in

    ch_out_power = np.mean(np.abs(ch_out) ** 2)
    ch_out = np.sqrt(sig_pwr) * (ch_out/ch_out_power) + noise

    return ch_out


def plot_const(mod):
    fig, ax = plt.subplots()
    plt.scatter(mod.real, mod.imag)
    plt.show()


def transmitter(M, num_sym, mod_type, const_plot, sig_pwr):
    # print("This is M and num_sym", M, num_sym)
    index = np.random.randint(0, M, (1, num_sym))
    alphabet = create_constellation(mod_type, M)
    mean_amp_const = np.sqrt(np.mean(alphabet * np.conj(alphabet)))
    norm_alp = alphabet * np.sqrt(sig_pwr) / mean_amp_const

    # if const_plot:
    #     plot_const(norm_alp)

    ch_input = norm_alp[index]
    return ch_input


def create_constellation(mod_type, M):
    if mod_type == "QAM":
        if M == 16 or M == 64:
            m_I = np.sqrt(M)

            I_comp = np.arange(m_I)
            I_comp = I_comp - I_comp.mean()

            Q_comp = I_comp.reshape(np.prod(I_comp.shape), 1)

            qam_alphabet = I_comp + 1j * Q_comp
            qam_alphabet = qam_alphabet.reshape(np.prod(qam_alphabet.shape))

            return qam_alphabet

        elif M == 32:
            const = np.array([1 + 1j, 3 + 1j, 5 + 1j, 1 + 3j, 3 + 3j, 5 + 3j, 1 + 5j, 3 + 5j])
            const = np.append(const, -const)
            qam_alphabet = np.append(const, np.conj(const))

            return qam_alphabet

        else:
            raise Exception("M is not valid")

    elif mod_type == "PSK":

        phase = np.linspace(0, 2 * np.pi, num=M, endpoint=False)
        psk_alphabet = np.cos(phase) + 1j * np.sin(phase)

        return psk_alphabet
    else:
        raise Exception("Modulation type not valid")

def variable_sweep(mod_ord, num_samples, rotation, modulation, SNR_dB, sig_pwr):
    index = 0
    row_t = len(mod_ord)*len(rotation)*len(SNR_dB)
    input_tensor = np.array(np.zeros((row_t,num_samples), dtype='cdouble'))
    labels = []
    print("input_tensor", input_tensor.shape)
    for i in mod_ord:
        # print('This is mod_ord', i)
        for j in rotation:
            for k in SNR_dB:
                # print("This is i, j, k",i,j,k)
                tr_sym = transmitter(i, num_samples, modulation, plot_const_diag, sig_pwr)
                rec_sym = channel(k, sig_pwr, tr_sym, j)
                # rec_sym_re = np.real(rec_sym)
                # rec_sym_ima = np.imag(rec_sym)
                current_sample = rec_sym
                # print("Current_sample: ", current_sample.shape)
                input_tensor[index,:] = current_sample
                index += 1
                labels.append("QAM" + str(i))
    input_tensor_real = np.real(input_tensor)
    input_tensor_imag = np.imag(input_tensor)
    # return input_tensor_real, input_tensor_imag, labels
    return input_tensor, labels



if __name__ == "__main__":
    SNR_dB = np.arange(0, 21, 1)
    sig_pwr = 1
    mod_ord = np.array([16,32,64])
    num_samples = 1500
    modulation = "QAM"

    rotation = np.arange(0, 90.9, 0.9)
    # print(rotation)


    plot_const_diag = True

    num_samples = 3 # TODO
    # in_real, in_imag, labels = variable_sweep(mod_ord, num_samples, rotation, modulation,SNR_dB, sig_pwr)
    # print("real", in_real)
    # print()
    # print("imag", in_imag)
    # print("labels", labels[-50:-1])

    # print("This are the first entries of the in_tensor output: ", in_tensor.shape)
    # print(in_tensor)

    # tr_pow = np.mean(np.abs(tr_sym) ** 2)
    # rec_pow = np.mean(np.abs(rec_sym) ** 2)


    # plot_const(rec_sym)
input_tensor,labels = variable_sweep(mod_ord, num_samples, rotation, modulation,SNR_dB, sig_pwr)
brk_tr = round(input_tensor.shape[0]*.8)
brk_tst = round(input_tensor.shape[0]*.2)
train_images = input_tensor[:brk_tr,:]
test_images = input_tensor[-brk_tst:-1,:]

train_labels = labels[:brk_tr]
test_labels = labels[-brk_tst:-1]

print("train_images, test_images, train_labels, test_labels", len(train_images), len(test_images), len(train_labels), len(test_labels))
print(train_images[0])
print(train_labels[0])
X_train = train_images.reshape(train_images.shape[0], img_cols, img_rows, 1)
X_test = test_images.reshape(test_images.shape[0], img_cols, img_rows, 1)

model = models.Sequential()
model.add(layers.Conv2D(3, 3, activation='relu', input_shape=(1, 100, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(3, 3, activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(3, 3, activation='relu'))

