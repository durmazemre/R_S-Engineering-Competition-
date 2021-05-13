import data_io_ingestion as io
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy import interpolate
import sys

from myutilities import get_rrc_pulse
#np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

SIGNALS_DIR = "../Public_Data/"
SAMPLING_RATE = 100e6 # TODO this is also in the myutilities.py file

def symbol_rate_detection(sigI, sigQ): # NOT COMPLETE
    if False:
        tmp_s_rate = 100 # Hz
        tmp_Ts = 1 # second(s)
        tmp_endtime = tmp_Ts*100
        tmp_times = np.arange(0, tmp_endtime, 1/tmp_s_rate)
        tmp_signal = np.array([np.sin(2*np.pi*x/tmp_Ts) for x in tmp_times]) 
        #plt.plot(tmp_times, tmp_signal)
        #plt.show()
        tmp_ft = np.fft.fft(tmp_signal)
        freqs = np.linspace(0, tmp_s_rate, len(tmp_signal), endpoint=False)
        plt.plot(freqs, np.abs(tmp_ft), 'r')
        plt.plot(freqs, np.angle(tmp_ft), 'b')
        plt.show()

    sigCmplx = sigI + 1j*sigQ
    sigCmplx = np.abs(sigCmplx)
    # sigCmplx = sigCmplx**2
    # TODO TODO TODO
    freqs, ft_sigCmplx = signal.periodogram(sigCmplx, SAMPLING_RATE)
    # negative_indices = np.arange(len(freqs) - 1, len(freqs)//2, -1) # TODO make it more reliable
    # positive_indices = np.arange(1, len(freqs)//2) # TODO make it more reliable
    # ft_sigCmplx[positive_indices]
    # ft_sigCmplx[negative_indices]
    # ft_sigCmplx = np.fft.fft(sigCmplx)
    # freqs = np.linspace(0, SAMPLING_RATE, len(sigCmplx), endpoint=False)
    # freqs_MHz = freqs/1e6
    print("Results")
    peak = np.argmax(ft_sigCmplx)
    peak2 = np.argmax(ft_sigCmplx[:len(ft_sigCmplx)//2])
    print(peak, peak2)
    print(freqs[peak], freqs[peak2])
    print(freqs[peak]/1e6, freqs[peak2]/1e6, " MHz")

    plt.plot(freqs, ft_sigCmplx, 'r')
    # plt.plot(freqs_MHz, np.angle(ft_sigCmplx), 'b')
    plt.show()

    return None # TODO

def sampling(mf_sigI, mf_sigQ, Ts, nos, throw_out_fraction):
    ######
    # INIT
    ######
    START = 2 # we will only sample in range [START, end-START]
    LEN = len(mf_sigI)
    SAMPLES_COUNT = int(LEN/nos) - 2*START # TODO
    I_samples = np.empty(SAMPLES_COUNT)
    Q_samples = np.empty(SAMPLES_COUNT)
    sample_times = np.empty(SAMPLES_COUNT)
    interp_mf_sigI = interpolate.interp1d(range(LEN), mf_sigI)
    interp_mf_sigQ = interpolate.interp1d(range(LEN), mf_sigQ)
    def sampler(i, offset):
        center = (START+i+offset)*nos
        sample_times[i] = center
        I_samples[i] = interp_mf_sigI(center)
        Q_samples[i] = interp_mf_sigQ(center)
    def error_detector(i, offset):
        SHIFT = max(nos/2, 1)
        if False: # early late
            center = (START+i+offset)*nos
            first_term = interp_mf_sigI(center + SHIFT) - interp_mf_sigI(center - SHIFT)
            second_term = interp_mf_sigQ(center + SHIFT) - interp_mf_sigQ(center - SHIFT)
            return (first_term + second_term)/2
        else: # Gardner
            center = (START+i+offset)*nos
            first_term = interp_mf_sigI(center + SHIFT) - interp_mf_sigI(center - SHIFT)
            first_term *= I_samples[i] # TODO normalize somehow?
            second_term = interp_mf_sigQ(center + SHIFT) - interp_mf_sigQ(center - SHIFT)
            second_term *= Q_samples[i] # TODO normalize somehow?
            return first_term + second_term
    def loop_filter(disc):
        if not hasattr(loop_filter, "I"):
            loop_filter.I = 0
        # TODO make these depend on nos?
        Kp = 1.6
        Ki = 0.01
        P = Kp*disc
        loop_filter.I += Ki*disc # TODO Ki*disc*dt?
        # TODO clamp I
        # D = -Kd*disc/dt
        return P + loop_filter.I # + D

    ###############
    # SAMPLING LOOP
    ###############
    offset_array = np.empty(SAMPLES_COUNT) # DEBUG
    offset = 0.5 # initialize
    for i in range(SAMPLES_COUNT):
        sampler(i, offset)
        disc = error_detector(i, offset)
        error = loop_filter(disc)
        offset += error
        offset_array[i] = offset # DEBUG
        # print("filter.I", loop_filter.I, "error", error)

    #########
    # RESULTS
    #########
    print("offset mean", np.mean(offset_array), "\n"
          "offset var", np.var(offset_array), "\n"
          "offset [max, min]", [max(offset_array), min(offset_array)])
    #print("sample_times[0]", sample_times[0])
    tmp_diff_sample_times = [sample_times[x] - sample_times[x-1] for x in range(1, len(sample_times))]
    # print("diff sample_times", "min", min(tmp_diff_sample_times), "max", max(tmp_diff_sample_times))
    print("sample period mean", np.mean(tmp_diff_sample_times), "\n"
          "sample period var", np.var(tmp_diff_sample_times), "\n"
          "sample period [max, min]", [max(tmp_diff_sample_times), min(tmp_diff_sample_times)])

    # plt.plot(mf_sigI, color='orange')
    # plt.scatter(I_sample_times, I_samples, c='orange')
    # plt.scatter(sample_times, Q_samples, c='cyan')
    # plt.show()
    THROW_OUT = int(SAMPLES_COUNT*throw_out_fraction)
    print("THROWING OUT", THROW_OUT, "out of", SAMPLES_COUNT, "samples")
    I_results = I_samples[THROW_OUT:]
    Q_results = Q_samples[THROW_OUT:]
    # plt.scatter(I_results, Q_results)
    # plt.show()
    return I_results, Q_results


def matched_filtering(sig, symbol_rate, length):
    #symbol_rate = 1e6*data_characteristics["Symbol Rate (MHz)"][SIG_INDEX]
    #symbol_rate = 17.82e6
    Ts = 1/symbol_rate # symbol period
    nos = SAMPLING_RATE/symbol_rate # oversampling factor

    # IQ signals
    FULL_LEN = length*2
    sigI = sig[0:FULL_LEN:2]
    sigQ = sig[1:FULL_LEN+1:2]
    LEN = len(sigI)
    # plt.plot(sigI, color='red', marker='+', ls='-')
    # plt.plot(sigQ, color='blue', marker='+', ls='-')
    # plt.show()

    # generate RRC pulse
    rrc_sig, times = get_rrc_pulse(symbol_rate=symbol_rate)
    # plt.plot(times*symbol_rate, rrc_sig, 'green')
    # plt.show()

    # matched filtering
    mf_sigI = np.convolve(sigI, rrc_sig, 'same') # TODO look into 'same' option
    mf_sigQ = np.convolve(sigQ, rrc_sig, 'same')
    # plt.plot(sigI, color='red')
    # plt.plot(mf_sigI, color='orange')
    # plt.plot(sigQ, color='blue')
    # plt.plot(mf_sigQ, color='cyan')
    # plt.show()
    return mf_sigI, mf_sigQ, Ts, nos

def get_symbol_rate(SIG_INDEX, data_characteristics):
    symbol_rate = data_characteristics["Symbol Rate (MHz)"][SIG_INDEX]
    if symbol_rate is None:
        dictionary = {5:7.79, 6:15.66, 7:15.75, 8:8.23, 9:5, 10:20.0,
                      11:17.82, 12:12.05, 13:33.2, 14:5.0, 15:5.0}
        symbol_rate = dictionary[SIG_INDEX]
    symbol_rate *= 1e6
    return symbol_rate


def main_function(SIG_INDEX, length=2500, throw_out_fraction=0.3):
    print("SIG_INDEX: ", SIG_INDEX)
    input_data, data_characteristics = io.inventory_data(SIGNALS_DIR, verbose=True)
    print("Length we are using: ", length)
    symbol_rate = get_symbol_rate(SIG_INDEX, data_characteristics)
    print("symbol rate: ", symbol_rate)
    sig = input_data[SIG_INDEX] # signal
    mf_sigI, mf_sigQ, Ts, nos = matched_filtering(sig, symbol_rate, length)
    print("nos", nos)
    I_results, Q_results = sampling(mf_sigI, mf_sigQ, Ts, nos, throw_out_fraction)
    return I_results, Q_results

if __name__ == '__main__':
    SIG_INDEX = 0
    I_results, Q_results = main_function(SIG_INDEX)
    plt.scatter(I_results, Q_results)
    plt.show()
