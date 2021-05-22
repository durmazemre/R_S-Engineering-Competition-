"""
Contains helper functions to generate artificial/synthetic noisy samples from a particular constellation.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans


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

    ch_out = np.sqrt(sig_pwr) * ch_out + noise

    return ch_out


def plot_const(mod):
    fig, ax = plt.subplots()
    plt.scatter(mod.real, mod.imag)
    plt.show()


def transmitter(M, num_sym, mod_type, const_plot, sig_pwr):
    index = np.random.randint(0, M, (1, num_sym))
    alphabet = create_constellation(mod_type, M)
    mean_amp_const = np.sqrt(np.mean(alphabet * np.conj(alphabet)))
    norm_alp = alphabet * np.sqrt(sig_pwr) / mean_amp_const

    if const_plot:
        plot_const(norm_alp)

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


if __name__ == "__main__":
    SNR_dB = None
    sig_pwr = 1
    mod_ord = 2
    num_samples = 50000
    modulation = "PSK"
    rotation = None  # in degrees
    plot_const_diag = True
    tr_sym = transmitter(mod_ord, num_samples, modulation, plot_const_diag, sig_pwr)
    rec_sym = channel(SNR_dB, sig_pwr, tr_sym, rotation)

    tr_pow = np.mean(np.abs(tr_sym) ** 2)
    rec_pow = np.mean(np.abs(rec_sym) ** 2)

    norm_sym = rec_sym / rec_pow

    rec_pow2 = np.mean(np.abs(norm_sym) ** 2)

    plot_const(rec_sym)
    plot_const(norm_sym / rec_pow)
