import numpy
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlate2d, find_peaks
import scipy
from mpl_toolkits.mplot3d import axes3d

fs = 20e6
batchsize = 2048
capture_interval = 0.01
folder = "../Messungen/Testmessungen"
file = "capture_2022-06-01_10-38-59_3750MHz_20MSps_2048S_10ms.dat"


if __name__ == "__main__":

    data = numpy.fromfile(open(f"{folder}/{file}"), dtype=numpy.complex64)
    zc_seq = numpy.load('zc_sequence.npy')

    corr = correlate(data, zc_seq, mode = "full")
    corr_max = numpy.max(corr)
    start_of_first_batch = numpy.argmax(corr>corr_max*0.1)
    print(f'The start of the batches is at {start_of_first_batch}')

    recieved_power = []
    number_of_batches = int( (len(data) - start_of_first_batch) / batchsize )
    batches = []
    for b in range(0,number_of_batches):
        lower = start_of_first_batch + batchsize * b
        upper = start_of_first_batch + batchsize * (b+1)
        #calculate the recieved power
        recieved_power.append(numpy.sum(abs(data[lower:upper]**2)))
        #seperate the batches
        batches.append(corr[lower:upper])
    #print(len(batches))
    #create a time axis
    time = numpy.arange(0,capture_interval*number_of_batches, capture_interval)
    #plot the recieved power over time
    plt.plot(time, recieved_power)
    plt.xlabel("Time in [s]")
    plt.ylabel("Power of detected signal")
    plt.ylim([0, numpy.max(recieved_power)*1.02])
    plt.show()

    h_meas = []
    b=0
    for batch in batches:
        batch_max = numpy.max(batch)
        xpeaks = find_peaks(abs(batch), 0.9 * batch_max, distance = 200)[0]
        ypeaks = []
        h=[]
        for peak in xpeaks:
            peak_ =peak+b*batchsize
            h.append(corr[peak_-10:peak_+118])
        h_meas.append( numpy.mean(h[1:], axis =0))
        b+=1
    #starting_peak = xpeaks[0]
    #h_meas = []
    #batch = []
    #for peak in xpeaks:
        #if peak +256 < starting_peak + 2048:
            #batch.append(corr[peak-10:peak+118])
        #else:
            #h_meas.append(numpy.mean(batch, axis = 0))
            #print(batch)
            #batch = []
            #starting_peak = peak
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
