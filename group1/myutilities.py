import numpy as np

SAMPLING_RATE = 100e6
ROLL_OFF = 0.22

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


def test():
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


