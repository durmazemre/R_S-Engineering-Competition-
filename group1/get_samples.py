import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy import interpolate
import sys
sys.path.append(os.path.abspath("../common/"))
import data_io_ingestion as io
from myutilities import get_rrc_pulse
#np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

SAMPLING_RATE = 100e6 # TODO this is also in the myutilities.py file

#def symbol_rate_detection(sig, debug_symbol_rate): # NOT COMPLETE
def symbol_rate_detection(sig):
    # # DEBUG PREAMBLE
    # SIG_INDEX = 0
    # print("SIG_INDEX: ", SIG_INDEX)
    # input_data, data_characteristics = io.inventory_data(SIGNALS_DIR, verbose=True)
    # sig = input_data[SIG_INDEX] # signal
    # debug_symbol_rate = get_symbol_rate(SIG_INDEX, data_characteristics)
    # print("symbol rate: ", debug_symbol_rate)

    # if False:
    #     tmp_s_rate = 100 # Hz
    #     tmp_Ts = 1 # second(s)
    #     tmp_endtime = tmp_Ts*100
    #     tmp_times = np.arange(0, tmp_endtime, 1/tmp_s_rate)
    #     tmp_signal = np.array([np.sin(2*np.pi*x/tmp_Ts) for x in tmp_times]) 
    #     #plt.plot(tmp_times, tmp_signal)
    #     #plt.show()
    #     tmp_ft = np.fft.fft(tmp_signal)
    #     freqs = np.linspace(0, tmp_s_rate, len(tmp_signal), endpoint=False)
    #     plt.plot(freqs, np.abs(tmp_ft), 'r')
    #     plt.plot(freqs, np.angle(tmp_ft), 'b')
    #     plt.show()

    tmp_length = len(sig)//2
    sigI = sig[0:tmp_length*2:2]
    sigQ = sig[1:tmp_length*2:2]


    sigCmplx = sigI + 1j*sigQ
    sigCmplx = np.abs(sigCmplx) # TODO
    # sigCmplx = sigCmplx**2
    freqs, ft_sigCmplx = signal.periodogram(sigCmplx, SAMPLING_RATE)
    # negative_indices = np.arange(len(freqs) - 1, len(freqs)//2, -1) # TODO make it more reliable
    # positive_indices = np.arange(1, len(freqs)//2) # TODO make it more reliable
    # ft_sigCmplx[positive_indices]
    # ft_sigCmplx[negative_indices]
    # ft_sigCmplx = np.fft.fft(sigCmplx)
    # freqs = np.linspace(0, SAMPLING_RATE, len(sigCmplx), endpoint=False)
    freqs_MHz = freqs/1e6

    print("Results")
    peak_index = np.argmax(ft_sigCmplx)
    peak2_index = np.argmax(ft_sigCmplx[:len(ft_sigCmplx)//2])
    peak = ft_sigCmplx[peak_index]
    peak2 = ft_sigCmplx[peak2_index]
    print(peak, peak2)
    print(freqs[peak_index]/1e6, freqs[peak2_index]/1e6, " MHz")

    # # nonzero_where = np.where(np.sum(signal_array, axis=0) >= 1)
    # for peak_fraction in [8, 2]:
    #     nonzero_where = np.nonzero(ft_sigCmplx > peak/peak_fraction)
    #     # print(nonzero_where)
    #     bandwidth_upper = np.max(freqs[nonzero_where])
    #     bandwidth_lower = np.min(freqs[nonzero_where])
    #     print("bandwidth", "(1/", peak_fraction, ")", (bandwidth_upper - bandwidth_lower)/1e6, "MHz")
    #     # print("using Carson formula, R = ", 1e-6*2*(bandwidth_upper - bandwidth_lower)/(1+0.22), "MHz")
    # plt.axvline(x=bandwidth_upper, color='cyan', linestyle=':')
    # plt.axvline(x=bandwidth_lower, color='cyan', linestyle=':')
    # if debug_symbol_rate is not None:
    #     plt.axvline(x=debug_symbol_rate, color='yellow')
    # plt.plot(freqs/1e6, np.angle(ft_sigCmplx), 'b')

    plt.plot(freqs, (ft_sigCmplx), 'r') # np.log?
    plt.show()

    result = freqs[peak_index]
    result = round(result/1e4)*1e4
    return int(result)

def sampling(mf_sigI, mf_sigQ, Ts, nos, throw_out_fraction):
    ######
    # INIT
    ######
    # START = 2 # we will only sample in range [START, end-START]
    START = 4 # we will only sample in range [START, end-START]
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
        try:
            sampler(i, offset)
        except ValueError as e: # TODO
            print(e)
            SAMPLES_COUNT = i
            break

        disc = error_detector(i, offset)
        error = loop_filter(disc)
        offset += error
        offset_array[i] = offset # DEBUG
        # print("offset: ", offset)
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


def matched_filtering(sigI, sigQ, symbol_rate, length):
    Ts = 1/symbol_rate # symbol period
    nos = SAMPLING_RATE/symbol_rate # oversampling factor
    LEN = len(sigI)
    assert LEN == len(sigQ)

    # generate RRC pulse
    rrc_sig, times = get_rrc_pulse(symbol_rate=symbol_rate)
    #plt.plot(times*symbol_rate, rrc_sig, 'green')
    #plt.show()

    # matched filtering
    mf_sigI = np.convolve(sigI, rrc_sig, 'same') # TODO look into 'same' option
    mf_sigQ = np.convolve(sigQ, rrc_sig, 'same')
    plt.plot(sigI, color='red')
    plt.plot(mf_sigI, color='orange')
    plt.plot(sigQ, color='blue')
    plt.plot(mf_sigQ, color='cyan')
    plt.show()
    return mf_sigI, mf_sigQ, Ts, nos


# def get_symbol_rate(SIG_INDEX, data_characteristics):
#     symbol_rate = data_characteristics["Symbol Rate (MHz)"][SIG_INDEX]
#     if symbol_rate is None:
#         dictionary = {5:7.79, 6:15.66, 7:15.75, 8:8.23, 9:5, 10:20.0,
#                       11:17.82, 12:12.05, 13:33.2, 14:5.0, 15:5.0}
#         symbol_rate = dictionary.get(SIG_INDEX, None)
#     if symbol_rate is not None:
#         symbol_rate *= 1e6
#     return symbol_rate


#def get_samples(SIG_INDEX, length=25000, throw_out_fraction=0.3):
#    print("SIG_INDEX: ", SIG_INDEX)
#    input_data, data_characteristics = io.inventory_data(SIGNALS_DIR, verbose=True)
#    print("Length we are using: ", length)
#    sig = input_data[SIG_INDEX] # signal
#    symbol_rate = debug_symbol_rate = get_symbol_rate(SIG_INDEX, data_characteristics)
#    print("symbol rate: ", debug_symbol_rate)
# def get_samples(sig, length=25000, throw_out_fraction=0.3):
def get_samples(sig, length, throw_out_fraction=0.3):
    symbol_rate = symbol_rate_detection(sig)

    # IQ signals
    sigI = sig[0:length*2:2]
    sigQ = sig[1:length*2:2]
    # plt.plot(sigI, color='red', marker='+', ls='-')
    # plt.plot(sigQ, color='blue', marker='+', ls='-')
    # plt.show()

    mf_sigI, mf_sigQ, Ts, nos = matched_filtering(sigI, sigQ, symbol_rate, length)

    print("nos", nos)
    I_results, Q_results = sampling(mf_sigI, mf_sigQ, Ts, nos, throw_out_fraction)

    return I_results, Q_results

if __name__ == '__main__':
    input_data = io.get_set_two()
    # print("Length we are using: ", length)
    for i in range(0, len(input_data)):
        SIG_INDEX = i
        print("SIG_INDEX: ", SIG_INDEX)
        sig = input_data[SIG_INDEX] # signal
        print(len(sig)/1e3, "thousand samples")
        # length = int(min(len(sig), 100e3)//2)
        # length = len(sig)//2
        # get_samples(sig, length)
        # I_results, Q_results = get_samples(sig, length)
        # plt.scatter(I_results, Q_results)
        # plt.show()


