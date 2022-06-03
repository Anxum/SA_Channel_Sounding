import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, find_peaks
from scipy.stats import norm


folder = "../Messungen/Testmessungen"
file = "capture_2022-06-01_10-38-59_3750MHz_20MSps_2048S_10ms.dat"

def linear_interpolation_x(y,y1,y0,x1,x0):
    if not y0 >= y >= y1 and not y0 <= y <= y1:
        print("Warning: y should be between y0 and y1")

    return x0+(( (y-y0) * (x1-x0) )/(y1-y0))

def confidence_interval(data, percentage):
    quantile = (percentage + 1)/2
    mean = np.mean(data)
    var = np.var(data)
    n = len(data)
    sigma = norm.ppf(quantile)
    HIB = sigma * np.sqrt(var/n)
    return mean, np.array([mean - HIB, mean + HIB])


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

    data = np.fromfile(open(f"{folder}/{file}"), dtype=np.complex64)
    zc_seq = np.load('zc_sequence.npy')

    corr = correlate(data, zc_seq, mode = "full")
    corr_max = np.max(corr)
    start_of_first_batch = np.argmax(corr>corr_max*0.1)

    recieved_power = []
    number_of_batches = int( (len(data) - start_of_first_batch) / batchsize )
    batches = []
    for b in range(0,number_of_batches):
        lower = start_of_first_batch + batchsize * b
        upper = start_of_first_batch + batchsize * (b+1)
        #calculate the recieved power
        recieved_power.append(np.sum(abs(data[lower:upper]**2)))
        #seperate the batches
        batches.append(corr[lower:upper])

    #create a time axis
    time = np.arange(0,capture_interval*number_of_batches, capture_interval)
    #plot the recieved power over time
    plt.figure(num="P(t)")
    plt.plot(time, recieved_power)
    plt.xlabel("Time in [s]")
    plt.ylabel("Power of detected signal")
    plt.title("Recieved signal power over time")
    plt.ylim([0, np.max(recieved_power)*1.02])
    #plt.show()

    h_meas = []
    b=0
    for batch in batches:
        batch_max = np.max(batch)
        xpeaks = find_peaks(abs(batch), 0.9 * batch_max, distance = 200)[0]
        ypeaks = []
        h=[]
        for peak in xpeaks:
            peak_ =peak+b*batchsize
            h.append(corr[peak_-10:peak_+118])
        h_meas.append( np.mean(h[1:], axis =0))
        b+=1

    T = [] # timevariant transferfunction
    for h in h_meas:
        T.append(abs(np.fft.fftshift(np.fft.fft(h))))

    #for t in T:
        #plt.plot(abs(t))
    #plt.show()

    #B_coh
    B_coh_50 = []
    B_coh_90 = []
    for t in range(len(T)):
        freq_corr_function = np.fft.fftshift( correlate(T[t],T[t], mode = "full") )
        delta_f = 2*(fs/10**6)/len(freq_corr_function)
        freq_corr_max = abs( np.max(freq_corr_function) )

        x1 = ( np.argmax( abs(freq_corr_function) < 0.5 * freq_corr_max ))
        x0 = x1 - 1
        B_coh_50.append( linear_interpolation_x(0.5 * freq_corr_max, abs(freq_corr_function[x1]), abs(freq_corr_function[x0]), x1, x0) * delta_f )
        #print(f"x0 = {x0}, x1 = {x1}, y0 = {abs(freq_corr_function[x0])}, y1 = {abs(freq_corr_function[x1])}, y = {0.5 * freq_corr_max}, x = {linear_interpolation_x(0.5 * freq_corr_max, abs(freq_corr_function[x1]), abs(freq_corr_function[x0]), x1, x0)}")

        x1 = ( np.argmax( abs(freq_corr_function) < 0.9 * freq_corr_max ))
        x0 = x1 - 1
        B_coh_90.append( linear_interpolation_x(0.9 * freq_corr_max, abs(freq_corr_function[x1]), abs(freq_corr_function[x0]), x1, x0) * delta_f )

    plt.figure(num="B_coh(t)")
    plt.plot(time, B_coh_50, "-b", label = 'B_coh_50%(t)' )
    plt.plot(time, B_coh_90, "-r", label = 'B_coh_90%(t)' )
    plt.xlabel("Time in [s]")
    plt.ylabel("B_coh in [MHz]")
    plt.title("Coherence Bandwidths over time")
    plt.legend(loc="best")

    B_coh_50_mean, B_coh_50_conf_interval_999 = confidence_interval(B_coh_50, 0.999)
    B_coh_90_mean, B_coh_90_conf_interval_999 = confidence_interval(B_coh_90, 0.999)
    print(f'99.9% of values of B_coh_50 lie in between {B_coh_50_conf_interval_999}[Mhz]')
    print(f'99.9% of values of B_coh_90 lie in between {B_coh_90_conf_interval_999}[MHz]')

    #T_coh

    T = np.swapaxes(T,0,1)
    T_coh_50 = []
    T_coh_90 = []
    resolution_in_time = 1 #100ms
    batches_per_value = int(resolution_in_time/capture_interval)
    for t in range (0, number_of_batches, batches_per_value):
        T_coh_50_f = []
        T_coh_90_f = []
        for f in range(len(T)):
            time_corr_function = np.fft.fftshift( correlate(T[f][t:t+batches_per_value],T[f][t:t+batches_per_value], mode = "full") )

            delta_t = 2*resolution_in_time/len(time_corr_function)
            time_corr_max = abs( np.max(time_corr_function) )

            x1 = ( np.argmax( abs(time_corr_function) < 0.5 * time_corr_max ))
            x0 = x1 - 1
            T_coh_50_f.append(linear_interpolation_x(0.5 * time_corr_max, abs(time_corr_function[x1]), abs(time_corr_function[x0]), x1, x0) * delta_t )

            x1 = ( np.argmax( abs(time_corr_function) < 0.9 * time_corr_max ))
            x0 = x1 - 1
            T_coh_90_f.append( linear_interpolation_x(0.9 * time_corr_max, abs(time_corr_function[x1]), abs(time_corr_function[x0]), x1, x0) * delta_t )
        T_coh_50.append(np.mean(T_coh_50_f))
        T_coh_90.append(np.mean(T_coh_90_f))

    plt.figure()
    plt.plot(T_coh_50, "-b", label = 'T_coh_50%(t)' )
    plt.plot(T_coh_90, "-r", label = 'T_coh_90%(t)' )

    plt.show()
