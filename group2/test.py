"""
Used for the qualifers round and challenge 2 of the finals round.
This is the "main" file to run the group1 function and the group2 function in the correct order to solve the task.
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

import data_generation

sys.path.append(os.path.abspath("../group1/"))
sys.path.append(os.path.abspath("../common/"))
import data_io_ingestion as io

import get_samples
import functions


# input_data = io.get_set_one() # 19 signals
#input_data = io.get_set_two() # 5 signals
#SIG_INDEX = 0
#sig = input_data[SIG_INDEX] # signal

# SAMPLING
#length = int(min(len(sig), 200e3)//2) # max value is 200000 i.e 400000//2
#throw_out_fraction = 0.3 # should be between 0 and 1
#I_results, Q_results = get_samples.get_samples(sig, length, throw_out_fraction)

SNR_dB = 17
sig_pwr = 1
mod_ord = 64
num_samples = 50000
modulation = "QAM"
rotation = None  # in degrees
plot_const_diag = True
tr_sym = data_generation.transmitter(mod_ord, num_samples, modulation, plot_const_diag, sig_pwr)
rec_sym = data_generation.channel(SNR_dB, sig_pwr, tr_sym, rotation)

plt.scatter(np.real(rec_sym), np.imag(rec_sym))
plt.show()

# Normalized variance

# REST
#sys.exit(0)
print("Part1 over")

est_mod_type = functions.detect_mod_type(np.real(rec_sym), np.imag(rec_sym))

#est_mod_type = "Not_known"
print(est_mod_type)

centers_mtx, est_mod_order, est_mod_type = functions.detect_mod_order(I_results, Q_results, est_mod_type)

#print(est_mod_order)
#print(est_mod_type)





# SIG_INDEX      Symbol Rate    Modulation
###########################################
# 0                  5          BPSK
# 1              21.26          QPSK
# 2                 25         QAM16
# 3              13.11         QAM32
# 4              30.73         QAM64
# *5               7.79         BPSK
# *6              15.66        32QAM
# 
# AWGN
# *7              15.75        16QAM
# *8               8.23         QPSK
# 9                   5         BPSK
# *10              20.0        64QAM
# 
# PHASE DRIFT
# *11             17.82         QPSK
# 12              12.05         QAM64
# *13              33.2         32QAM
# *14               5.0         BPSK
# 
# CW
# ***15             5.0         BPSK
# 16               1.52        QAM16
# ?***17           31.99         None
# ?***18           18.32        QPSK
# 
# 17 --> maybe 10 MHz?
# 18 --> maybe 9.2 MHz?
