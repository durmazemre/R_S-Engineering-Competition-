import data_io_ingestion as io
import matplotlib.pyplot as plt
import numpy as np

SIGNALS_DIR = "../Public_Data/"
input_data, data_characteristics = io.inventory_data(SIGNALS_DIR, verbose=True)
SAMPLING_RATE = 100e6
ROLL_OFF = 0.22
SIG_INDEX = 2

sig = input_data[SIG_INDEX] # signal
symbol_rate = 1e6*data_characteristics["Symbol Rate (MHz)"][SIG_INDEX]
Ts = 1/symbol_rate # symbol period
nos = int(SAMPLING_RATE/symbol_rate) # oversampling factor

# IQ signals
LEN = nos*50
sigI = sig[1:LEN:2]
sigQ = sig[2:LEN:2]
plt.plot(sigI, 'r')
plt.plot(sigQ, 'b')
plt.show()

# generate RRC pulse
print("symbol rate: ", symbol_rate)
rrc_sig, times = get_rrc_pulse(symbol_rate=symbol_rate)
#plt.plot(rrc_sig, 'green')
#plt.plot(times*symbol_rate, rrc_sig, 'green')
#plt.show()

# matched filtering
mf_sigI = np.convolve(sigI, rrc_sig, 'same')
mf_sigQ = np.convolve(sigQ, rrc_sig, 'same')
plt.plot(sigI, 'orange')
plt.plot(mf_sigI, 'red')
plt.plot(sigQ, 'cyan')
plt.plot(mf_sigQ, 'blue')
plt.show()

# sampling
# TODO


def get_rrc_pulse(symbol_rate=5e6, rolloff=ROLL_OFF, filterspan=20, sampling_rate=SAMPLING_RATE):
    Ts = 1/symbol_rate
    rrc_range = Ts*filterspan/2
    times = np.arange(-rrc_range, rrc_range, 1/sampling_rate);
    return rrc(times, Ts, rolloff), times


def rrc(times, Ts, rolloff):
    # TODO is this scaling correct/appropriate?
    scale = 1/nos
    scale *= 1/symbol_rate # apparently the energy of this signal equals the symbol rate if I don't scale it
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


def test()
    rc_sig = np.convolve(rrc_sig, rrc_sig, 'same')
    plt.plot(rc_sig, 'greenyellow')
    #plt.plot(times*symbol_rate, rc_sig, 'greenyellow')
    plt.show()
    pulse = [rc_sig/20, rrc_sig][2]
    color = ['gray', 'black'][2]
    msg = [1, 1, 1, -1, 1, -1, 1, -1]
    nos = int(Ts*sampling_rate) # oversampling factor
    pulse_msg = np.zeros((len(msg) - 1)*nos+len(pulse), 'float32')
    for i in range(len(msg)):
        pulse_msg[i*nos:i*nos+len(pulse)] += msg[i]*pulse
    plt.plot(pulse_msg, color)
    plt.show()
    ###########
    s_rate = 100e6
    s_period = 1/s_rate
    # rrc_sig[times[len(times)//2]]
    sum_rrc_sig = sum(rrc_sig**2)
    print((s_period*sum_rrc_sig))
    len(rrc_sig)



