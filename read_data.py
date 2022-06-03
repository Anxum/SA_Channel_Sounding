import numpy
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlate2d, find_peaks
import scipy


folder = "../Messungen/Testmessungen"
file = "capture_2022-06-01_10-38-59_3750MHz_20MSps_2048S_10ms.dat"

def linear_interpolation_x(y,y1,y0,x1,x0):
    if not y0 >= y >= y1:
        print("Warning: y should be between y0 and y1")

    return x0+(( (y-y0) * (x1-x0) )/(y1-y0))

def read_meta_data(filename):
    str = filename.split('_')
    fc_ = int( str[3][0:str[3].find('MHz')] ) * 10**6
    fs_ = int( str[4][0:str[4].find('MSps')] ) * 10**6
    batchsize_ = int( str[5][0:str[5].find('S')] )
    capture_interval_ = int( str[6][0:str[6].find('ms')] ) * 10**-3
    print(f'Date of measurement = {str[1]}')
    print(f'Time of measurement = {str[2]}')
    print(f'fc = {str[3]}, fs = {str[4]}, batchsize = {str[5]}, capture interval = {str[6][:-3]}')
    return fc_, fs_, batchsize_, capture_interval_


if __name__ == "__main__":
    fc, fs, batchsize, capture_interval = read_meta_data(file)

    data = numpy.fromfile(open(f"{folder}/{file}"), dtype=numpy.complex64)
    zc_seq = numpy.load('zc_sequence.npy')

    corr = correlate(data, zc_seq, mode = "full")
    corr_max = numpy.max(corr)
    start_of_first_batch = numpy.argmax(corr>corr_max*0.1)

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
    T = [] # timevariant transferfunction
    for h in h_meas:
        T.append(abs(numpy.fft.fftshift(numpy.fft.fft(h))))

    #for t in T:
        #plt.plot(abs(t))
    #plt.show()
    B_coh_50 = []
    B_coh_90 = []
    for t in range(len(T)):
        freq_corr_function = numpy.fft.fftshift( correlate(T[t],T[t], mode = "full") )
        delta_x = 2*fs/len(freq_corr_function)
        freq_corr_max = abs( numpy.max(freq_corr_function) )

        x1 = ( numpy.argmax( abs(freq_corr_function) < 0.5 * freq_corr_max ))
        x0 = x1 - 1
        B_coh_50.append( linear_interpolation_x(0.5 * freq_corr_max, abs(freq_corr_function[x1]), abs(freq_corr_function[x0]), x1, x0) * delta_x )
        #print(f"x0 = {x0}, x1 = {x1}, y0 = {abs(freq_corr_function[x0])}, y1 = {abs(freq_corr_function[x1])}, y = {0.5 * freq_corr_max}, x = {linear_interpolation_x(0.5 * freq_corr_max, abs(freq_corr_function[x1]), abs(freq_corr_function[x0]), x1, x0)}")

        x1 = ( numpy.argmax( abs(freq_corr_function) < 0.9 * freq_corr_max ))
        x0 = x1 - 1
        B_coh_90.append( linear_interpolation_x(0.9 * freq_corr_max, abs(freq_corr_function[x1]), abs(freq_corr_function[x0]), x1*delta_x, x0*delta_x) )


    plt.plot(time, B_coh_50)
    plt.plot(time, B_coh_90)
    plt.show()
    #h_mean = numpy.mean(h_meas, axis = 0)
    #H = scipy.fft(h_mean)
    #H = numpy.fft.fftshift(H)

    #plt.plot(freq,abs(H))
    #plt.show()
    #plt.plot(abs(h))
    #plt.show()

    #print(corr)
