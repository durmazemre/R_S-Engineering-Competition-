# 64 QAM
cnstln = [1+j, 2+j, 1+2j, ...] # one quadrant
cnstln = np.concatenate(np.conjugate(constln), cnstln)
cnstln = np.concatenate(-constln, cnstln)
cnstln = cnstln/get_energy(cnsltn)

# energy
samples = # = get_samples.main_function(...) converted to complex samples
energy = get_energy(samples)
samples = samples/np.sqrt(energy)


def get_energy(samples):
    return np.mean(np.abs(samples)**2)

def rotate(arrayI, arrayQ, theta):
    array = arrayI + 1j*arrayQ
    return array * (np.cos(theta) + j*np.sin(theta))


def generate_samples(cnstln, noise, sample_count):
    each_count = sample_count/len(cnstln)
    result_array = np.empty(sample_count)
    for mean in cnstln:
        for i in range(each_count):
            result_array = mean + noise*(randn(...) + 1j*randn(...))/np.sqrt(2)
    return result_array


def convert_to_image(samples):
    IMAGE_HEIGHT = 100
    IMAGE_WIDTH = 100

    # [..., QPSK, 16QAM, 32QAM, ...]
      [..., 0.01, 0.82, 0.25, ...]  = neural_network(samples)
