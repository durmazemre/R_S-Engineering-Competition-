"""
Used for qualifiers round and challenge 2 of the finals.
Does constellation recognition based on samples given as input.
"""
import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten

def detect_mod_type(real_sym,imag_sym):
    sym = real_sym + 1j * imag_sym
    power = np.sqrt(sym * np.conjugate(sym))
    ratio = np.amax(power) / np.amin(power)
    mag_sig = np.abs(sym)
    norm_var = np.var(mag_sig) / np.mean(mag_sig ** 2)

    if norm_var < 0.1: #7'den büyük ise 64qam olabilir!
        est_mod_type = "PSK"
    elif ratio > 3 and ratio < 7:
        est_mod_type = "Not_known"
    else:
        est_mod_type = "QAM"

    return  est_mod_type



def detect_mod_order(real_sym,imag_sym, est_mod_type):
    sym = real_sym + 1j*imag_sym
    amp = np.sqrt(sym * np.conjugate(sym))
    mean_amp = np.mean(amp)

    data_mtx = np.concatenate((np.array(real_sym, ndmin=2).T, np.array(imag_sym, ndmin=2).T), axis=1)

    if est_mod_type == "QAM":
        possible_orders = [64, 32, 16, 4, 2]
    elif est_mod_type == "PSK":
        possible_orders = [4,2]
    else:
        possible_orders = [32, 16, 4, 2]
    distortion_cont = np.array([])


    for i in possible_orders:
        centers_mtx, distortion = kmeans(data_mtx, i)
        distortion_cont = np.append(distortion_cont, distortion)

    distortion_cont2 = np.roll(distortion_cont, len(possible_orders) - 1)
    distortion_cont2[-1] = 0
    differences = np.absolute((distortion_cont - distortion_cont2) / distortion_cont)
    ind = np.argmax(differences)
    est_mod_ind = possible_orders[ind]

    if est_mod_ind < 16:
        est_mod_type = "PSK"
    else:
        est_mod_type = "QAM"

    centers_mtx, distortion = kmeans(data_mtx, est_mod_ind)
    return centers_mtx, est_mod_ind, est_mod_type
