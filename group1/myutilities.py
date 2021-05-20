import os
import sys
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath("../common/"))
import data_io_ingestion as io

SAMPLING_RATE = 100e6
ROLL_OFF = 0.22
SYNC = "010001000100001011101011111111001110100000111110110101011000000001101111010111010101001011111111111001111111101011010110"
SAMPLES_DIR = '../Public Finals Samples/'

def save_file(samplesI, samplesQ, filename):
    samples = samplesI + 1j*samplesQ
    path = SAMPLES_DIR + filename + '.npy'
    with open(path, 'wb') as f:
        np.save(f, samples)

def get_sync(M): # M is modulation order
    # return SYNC sequence in form of modulation symbols
    sync = np.fromstring(SYNC,'u1') - ord('0')
    bit_count = int(np.log2(M))
    input_data = io.get_set_two(cnstln=True)
    d = {2: 0, 4: 4, 16:1, 32:2, 64:3} # dictionary: M -> input_data index array
    k = d[M]
    cnstln = input_data[k]
    cnstln_r = cnstln[0::2]
    cnstln_i = cnstln[1::2]
    symbols = []
    for i in range(len(SYNC)//bit_count):
        tmp = sync[i*bit_count:(i+1)*bit_count]
        index = tmp.dot(2**np.arange(tmp.size)[::-1])
        cmplx = cnstln_r[index] + 1j*cnstln_i[index]
        symbols.append(cmplx)
    return symbols


def plotpsd(sigCmplx, details=True):
    freqs, ft_sigCmplx = signal.periodogram(sigCmplx, SAMPLING_RATE)
    freqs_MHz = freqs/1e6
    if details:
        peak_indices = largest_indices(ft_sigCmplx, 3)
        peaks = ft_sigCmplx[peak_indices]
        # print(peaks)
        print(freqs[peak_indices]/1e6, " MHz")
        # print(freqs[peak_index]/1e6, freqs[peak2_index]/1e6, " MHz")
    plt.plot(freqs, (ft_sigCmplx), 'r') # np.log?
    plt.show()

        # peak_index = np.argmax(ft_sigCmplx)
        # peak = ft_sigCmplx[peak_index]
        # print(peak)
        # print(freqs[peak_index]/1e6, " MHz")

# credit: https://stackoverflow.com/a/38884051
def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)

def get_rrc_sequence(samples, symbol_rate, rolloff=ROLL_OFF, filterspan=20, sampling_rate=SAMPLING_RATE):
    pulse = get_rrc_pulse(symbol_rate, rolloff, filterspan, sampling_rate)
    nos = sampling_rate/symbol_rate
    Ts = 1/symbol_rate
    for s in samples:
        s*pulse

def get_rrc_pulse(symbol_rate, rolloff=ROLL_OFF, filterspan=20, sampling_rate=SAMPLING_RATE):
    nos = sampling_rate/symbol_rate
    Ts = 1/symbol_rate
    rrc_range = Ts*filterspan/2
    times = np.arange(-rrc_range, rrc_range, 1/sampling_rate);
    return rrc(times, Ts, nos, rolloff), times

def rrc(times, Ts, nos, rolloff):
    # TODO is this scaling correct/appropriate?
    scale = 1/nos
    scale *= Ts # apparently the energy of this signal equals the symbol rate if I don't scale it
    iter_obj = (scale*rrc_impulse(t, Ts, rolloff) for t in times)
    # TODO specify count parameter as well for more efficiency
    return np.fromiter(iter_obj, 'float32')


def rrc_impulse(t, Ts, rolloff):
    if t == 0:
        return (1/Ts)*(1 + rolloff*(4/np.pi - 1))
    elif abs(t) == Ts/(4*rolloff):
        first_term = (1+2/np.pi)*np.sin(np.pi/(4*rolloff))
        second_term = (1-2/np.pi)*np.cos(np.pi/(4*rolloff))
        return (first_term + second_term)*rolloff/(Ts*np.sqrt(2))
    else:
        first_term = np.sin((1-rolloff)*np.pi*t/Ts)
        second_term = np.cos((1+rolloff)*np.pi*t/Ts)*4*rolloff*t/Ts
        denom = (1-(4*rolloff*t/Ts)**2)*np.pi*t/Ts
        return (1/Ts)*(first_term + second_term)/denom


def sampling2(mf_sigI, mf_sigQ, Ts, nos):
    LEN = len(mf_sigI)
    #if True:
    #for PHASE in range(0, min(2*round(nos), 10)):
    for PHASE in range(7, 12):
        #for PHASE in [5.5]:
        #for PHASE in np.arange(0, min(nos+1, 20), 0.5):
        print("PHASE", PHASE)
        # PHASE = 0
        interp_mf_sigI = interpolate.interp1d(range(LEN), mf_sigI)
        interp_mf_sigQ = interpolate.interp1d(range(LEN), mf_sigQ)

        sample_times = [PHASE + i*nos for i in range(0, int((LEN - PHASE)//nos))]
        xdata = interp_mf_sigI(sample_times)
        ydata = interp_mf_sigQ(sample_times)

        plt.title("Phase = " + str(PHASE))
        plt.scatter(xdata, ydata)
        plt.show()

        plt.plot(mf_sigI, color='orange')
        plt.scatter(sample_times, xdata, c='orange')
        plt.plot(mf_sigQ, color='cyan')
        plt.scatter(sample_times, ydata, c='cyan')
        plt.title("Phase = " + str(PHASE))
        plt.show()


# def test():
#     rc_sig = np.convolve(rrc_sig, rrc_sig, 'same')
#     plt.plot(rc_sig, 'greenyellow')
#     #plt.plot(times*symbol_rate, rc_sig, 'greenyellow')
#     plt.show()
#     pulse = [rc_sig/20, rrc_sig][2]
#     color = ['gray', 'black'][2]
#     msg = [1, 1, 1, -1, 1, -1, 1, -1]
#     nos = int(Ts*sampling_rate) # oversampling factor
#     pulse_msg = np.zeros((len(msg) - 1)*nos+len(pulse), 'float32')
#     for i in range(len(msg)):
#         pulse_msg[i*nos:i*nos+len(pulse)] += msg[i]*pulse
#         plt.plot(pulse_msg, color)
#         plt.show()
#         ###########
#     s_rate = 100e6
#     s_period = 1/s_rate
#     # rrc_sig[times[len(times)//2]]
#     sum_rrc_sig = sum(rrc_sig**2)
#     print((s_period*sum_rrc_sig))
#     len(rrc_sig)



# def DEBUGGING_CODE_FOR_symbol_rate_detection(sig):
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


    # negative_indices = np.arange(len(freqs) - 1, len(freqs)//2, -1) # TODO make it more reliable
    # positive_indices = np.arange(1, len(freqs)//2) # TODO make it more reliable
    # ft_sigCmplx[positive_indices]
    # ft_sigCmplx[negative_indices]
    # ft_sigCmplx = np.fft.fft(sigCmplx)

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
