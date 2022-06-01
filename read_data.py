import numpy
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlate2d, find_peaks
import scipy
from mpl_toolkits.mplot3d import axes3d

fs = 20e6


if __name__ == "__main__":
    f = numpy.fromfile(open('../Messungen/Testmessungen/capture_2022-06-01_10-38-59_3750MHz_20MSps_2048S_10ms.dat'), dtype=numpy.complex64)
    zc_seq = numpy.load('zc_sequence.npy')
    #print(len(f))
    data = f
    corr = correlate(data, zc_seq, mode = "valid")
    corr_max = numpy.max(corr)
    plt.plot(abs(corr))
    #plt.show()
    xpeaks = find_peaks(abs(corr), 0.01 * corr_max, distance = 200)[0]
    xpeaks = xpeaks
    ypeaks = []
    for peak in xpeaks:
        ypeaks.append(abs(corr[peak]))
    plt.scatter(xpeaks, ypeaks)
    plt.show()
    starting_peak = xpeaks[0]
    h_meas = []
    batch = []
    for peak in xpeaks:
        if peak +256 < starting_peak + 2048:
            batch.append(corr[peak-10:peak+118])
        else:
            h_meas.append(numpy.mean(batch, axis = 0))
            #print(batch)
            batch = [corr[peak-10:peak+118]]
            starting_peak = peak
    #print(h_meas)
    T = [] # timevariant transferfunction
    for h in h_meas:
        T.append(abs(numpy.fft.fftshift(numpy.fft.fft(h))))

    plt.plot(abs(T[round(len(T)/2)]))
    plt.show()

    freq_corr_function = correlate(T[round(len(T)/2)],T[round(len(T)/2)], mode = "full")
    #freq = numpy.fft.fftshift(numpy.fft.fftfreq(511, d = 1/fs))
    #print(phi_T)
    #freq_corr_function = phi_T[round(len(phi_T)/2)]
    #print(freq_corr_function)

    plt.plot(abs(freq_corr_function))
    plt.show()
    #h_mean = numpy.mean(h_meas, axis = 0)
    #H = scipy.fft(h_mean)
    #H = numpy.fft.fftshift(H)

    #plt.plot(freq,abs(H))
    #plt.show()
    #plt.plot(abs(h))
    #plt.show()

    #print(corr)
