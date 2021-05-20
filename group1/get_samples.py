import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy import interpolate
import sys
from myutilities import get_rrc_pulse
from myutilities import plotpsd
from myutilities import get_sync
import myutilities

sys.path.append(os.path.abspath("../common/"))
import data_io_ingestion as io
from data_io_ingestion import get_symbol_rate
#np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

SAMPLING_RATE = 100e6 # TODO this is also in the myutilities.py file

def lowpass(sig):
    order = 10
    Wn = 5e6
    btype='lowpass'
    sos = signal.butter(order, Wn, btype, fs=SAMPLING_RATE, output='sos')
    w, h = signal.sosfreqz(sos, fs=SAMPLING_RATE)
    plt.plot(w, h)
    # filtered = signal.sosfilt(sos, sig)
    filtered = signal.sosfilt(sos, sig)
    return filtered

def filtering(sig):
    f_remove = 40e6
    f_remove = 31.99e6
    Q=35
    b, a = signal.iirnotch(f_remove, Q, SAMPLING_RATE)
    return b, a
    # freq, h = signal.freqz(b, a, fs=SAMPLING_RATE)
    # plt.plot(freq, np.log(h))
    # plt.show()
    # signal.lfilter(b, a, sig)


def symbol_rate_detection(sig):
    tmp_length = len(sig)//2
    sigI = sig[0:tmp_length*2:2]
    sigQ = sig[1:tmp_length*2:2]

    sigCmplx = sigI + 1j*sigQ
    if False: # plot original spectrum # and perhaps modify original signal
        freqs, ft_sigCmplx0 = signal.periodogram(sigCmplx, SAMPLING_RATE)
        freqs_MHz = freqs/1e6
        plt.plot(freqs, (ft_sigCmplx0), 'r') # np.log?
        plt.show()

        if False: # modifications
            # b, a = filtering(sigCmplx)
            sigCmplx = lowpass(sigCmplx)
            # sigCmplx = signal.lfilter(b, a, sigCmplx)
            freqs, ft_sigCmplx = signal.periodogram(sigCmplx, SAMPLING_RATE)
            freqs_MHz = freqs/1e6
            plt.plot(freqs, (ft_sigCmplx), 'r') # np.log?
            plt.show()
            plt.plot(freqs, (ft_sigCmplx - ft_sigCmplx0), 'r') # np.log?
            plt.show()

    sigCmplx = np.abs(sigCmplx) # OR consider: sigCmplx = sigCmplx**2
    freqs, ft_sigCmplx = signal.periodogram(sigCmplx, SAMPLING_RATE)
    freqs_MHz = freqs/1e6

    if False: # plot new spectrum
        freqs, ft_sigCmplx = signal.periodogram(sigCmplx, SAMPLING_RATE)
        freqs_MHz = freqs/1e6
        plt.plot(freqs, (ft_sigCmplx), 'r') # np.log?
        plt.show()

    print("Results")
    peak_index = np.argmax(ft_sigCmplx)
    peak2_index = np.argmax(ft_sigCmplx[:len(ft_sigCmplx)//2])
    peak = ft_sigCmplx[peak_index]
    peak2 = ft_sigCmplx[peak2_index]
    print(peak, peak2)
    print(freqs[peak_index]/1e6, freqs[peak2_index]/1e6, " MHz")
    result = freqs[peak_index]
    result = round(result/1e4)*1e4
    return int(result)

def sampling(mf_sigI, mf_sigQ, Ts, nos, throw_out_fraction):
    ######
    # INIT
    ######
    START = 4 # we will only sample in range [START, end-START]

    LEN = len(mf_sigI)
    SAMPLES_COUNT = int(LEN/nos) - 2*START # TODO
    I_samples = np.empty(SAMPLES_COUNT)
    Q_samples = np.empty(SAMPLES_COUNT)
    sample_times = np.empty(SAMPLES_COUNT)
    upsample_factor = max(np.ceil(30/nos), nos)
    def upsample(mf_sigI, mf_sigQ):
        interp_mf_sigI = interpolate.interp1d(range(LEN), mf_sigI)
        interp_mf_sigQ = interpolate.interp1d(range(LEN), mf_sigQ)
        up_times = np.arange(0, LEN - 1, 1/upsample_factor)
        print(up_times, LEN)
        up_mf_sigI = interp_mf_sigI(up_times)
        up_mf_sigQ = interp_mf_sigQ(up_times)
        return up_mf_sigI, up_mf_sigQ

    up_mf_sigI, up_mf_sigQ = upsample(mf_sigI, mf_sigQ)
    def sampler(i, offset):
        if False: # interpolate version
            center = (START+i+offset)*nos
            sample_times[i] = center
            I_samples[i] = interp_mf_sigI(center)
            Q_samples[i] = interp_mf_sigQ(center)
        else: # upsample version
            center = (START+i+offset)*nos*upsample_factor
            sample_times[i] = center
            nearest = int(round(center))
            I_samples[i] = up_mf_sigI[nearest]
            Q_samples[i] = up_mf_sigQ[nearest]

    def error_detector(i, offset):
        SHIFT = max(nos/2, 1)
        if False: # early late
            center = (START+i+offset)*nos
            first_term = interp_mf_sigI(center + SHIFT) - interp_mf_sigI(center - SHIFT)
            second_term = interp_mf_sigQ(center + SHIFT) - interp_mf_sigQ(center - SHIFT)
            return (first_term + second_term)/2
        else: # Gardner
            if False: # inteperpolate version
                center = (START+i+offset)*nos*upsample_factor
                first_term = interp_mf_sigI(center + SHIFT) - interp_mf_sigI(center - SHIFT)
                first_term *= I_samples[i] # TODO normalize somehow?
                second_term = interp_mf_sigQ(center + SHIFT) - interp_mf_sigQ(center - SHIFT)
                second_term *= Q_samples[i] # TODO normalize somehow?
                return first_term + second_term
            else: # upsample version
                center = (START+i+offset)*nos*upsample_factor
                nearest_plus = int(round(center + SHIFT))
                nearest_minus = int(round(center - SHIFT))
                first_term = up_mf_sigI[nearest_plus] - up_mf_sigI[nearest_minus]
                first_term *= I_samples[i] # TODO normalize somehow?
                second_term = up_mf_sigQ[nearest_plus] - up_mf_sigQ[nearest_minus]
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
    # offset = 0.5 # initialize
    offset = 0.974 # TODO TODO TODO
    # offset = 0.5855 # TODO TODO TODO
    for i in range(SAMPLES_COUNT):
        # try:
        sampler(i, offset)
        disc = error_detector(i, offset)
        # except ValueError as e: # TODO
        #     print(e)
        #     SAMPLES_COUNT = i
        #     break
        error = loop_filter(disc)
        offset += error
        offset_array[i] = offset # DEBUG
        if i % 500 == 0:
            print("offset: ", offset)
        # print("filter.I", loop_filter.I, "error", error)

    #########
    # RESULTS
    #########
    THROW_OUT = int(SAMPLES_COUNT*throw_out_fraction)
    print("offset mean", np.mean(offset_array), "\n",
          "offset (thrown out) mean", np.mean(offset_array[THROW_OUT:]), "\n",
          "offset var", np.var(offset_array), "\n"
          "offset [max, min]", [max(offset_array), min(offset_array)])
    #print("sample_times[0]", sample_times[0])
    tmp_diff_sample_times = [sample_times[x] - sample_times[x-1] for x in range(1, SAMPLES_COUNT)]
    # print("diff sample_times", "min", min(tmp_diff_sample_times), "max", max(tmp_diff_sample_times))
    print("sample period mean", np.mean(tmp_diff_sample_times), "\n"
          "sample period var", np.var(tmp_diff_sample_times), "\n"
          "sample period [max, min]", [max(tmp_diff_sample_times), min(tmp_diff_sample_times)])

    # plt.plot(mf_sigI, color='orange')
    # plt.scatter(I_sample_times, I_samples, c='orange')
    # plt.scatter(sample_times, Q_samples, c='cyan')
    # plt.show()
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
    # plt.plot(sigI, color='red')
    # plt.plot(mf_sigI, color='orange')
    # plt.plot(sigQ, color='blue')
    # plt.plot(mf_sigQ, color='cyan')
    # plt.show()
    return mf_sigI, mf_sigQ, Ts, nos



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
    # I_results, Q_results = get_samples(sig, length)
    set_two = True
    SIG_INDEX = 1
    print("SIG_INDEX: ", SIG_INDEX)
    if set_two:
        input_data = io.get_set_two()
    else:
        input_data = io.get_set_one()
        # for x in range(0,5):
    #     print(input_data[x])
    #     print(len(input_data[x]))
    sig = input_data[SIG_INDEX] # signal
    print("full length (as IQ pairs)", len(sig)//2)

    I_results, Q_results = io.load_samples(SIG_INDEX)
    plt.scatter(I_results, Q_results)
    plt.show()

    # print("Length we are using: ", length)
    # for i in range(0, len(input_data)):
    # filtering(None)
    if False:
        symbol_rate = symbol_rate_detection(sig)
    if False:
        symbol_rate = get_symbol_rate(SIG_INDEX, set_two=set_two)
        print("symbol_rate", symbol_rate/1e6, " MHz")
        # length = int(min(len(sig), 100e3)//2)
        # length = int(min(len(sig), 200e3)//2)
        length = len(sig)

        # IQ signals
        sigI = sig[0:length*2:2]
        sigQ = sig[1:length*2:2]
        mf_sigI, mf_sigQ, Ts, nos = matched_filtering(sigI, sigQ, symbol_rate, length)
        if False:
            plotpsd(sigI + 1j*sigQ, details=False)
            plotpsd(np.abs(sigI + 1j*sigQ))
            plotpsd(mf_sigI + 1j*mf_sigQ, details=False)
            plotpsd(np.abs(mf_sigI + 1j*mf_sigQ))
        else:
            if False:
                plt.plot(sigI, color='red')
                plt.plot(mf_sigI, color='orange')
                # plt.plot(sigQ, color='blue')
                # plt.plot(mf_sigQ, color='cyan')
                plt.show()

        print("Ts", Ts, "nos", nos, "symbol_rate", symbol_rate/1e6, " MHz")
        I_results, Q_results = sampling(mf_sigI, mf_sigQ, Ts, nos, throw_out_fraction=0.0)
        plt.scatter(I_results, Q_results)
        plt.show()
        print("number of samples saved: ", len(I_results))
        myutilities.save_file(I_results, Q_results, "sig" + str(SIG_INDEX))



