import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("../group1/"))
import get_samples


SIG_INDEX = 0
length = 25000 # max value is 200000
throw_out_fraction = 0.3 # should be between 0 and 1
I_results, Q_results = get_samples.main_function(SIG_INDEX, length, throw_out_fraction)
plt.scatter(I_results, Q_results)
plt.show()



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
