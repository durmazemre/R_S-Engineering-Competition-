import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("../group1/"))
import get_samples
import data_io_ingestion as io
import functions

# GET SIGNAL
SIGNALS_DIR = "../Public_Data/"
input_data, data_characteristics = io.inventory_data(SIGNALS_DIR, verbose=True)
SIG_INDEX = 12
sig = input_data[SIG_INDEX] # signal

# SAMPLING
length = int(min(len(sig), 200e3)//2) # max value is 200000 i.e 400000//2
throw_out_fraction = 0.3 # should be between 0 and 1
I_results, Q_results = get_samples.get_samples(sig, length, throw_out_fraction)
plt.scatter(I_results, Q_results)
plt.show()

# REST
sys.exit(0)
print("Part1 over")

est_mod_type = functions.detect_mod_type(I_results, Q_results)

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
