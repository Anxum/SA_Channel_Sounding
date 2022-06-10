import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, find_peaks
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D


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
    print(f'fc = {fc_ * 10**-6}MHz, fs = {fs_ * 10**-6}MHz, batchsize = {batchsize_}S, capture interval = {capture_interval_ * 10**3}ms')
    return fc_, fs_, batchsize_, capture_interval_


if __name__ == "__main__":
    fc, fs, batchsize, capture_interval = read_meta_data(file)
    data = np.fromfile(open(f"{folder}/{file}"), dtype=np.complex64)
    zc_seq = np.load('zc_sequence.npy')
    # Todo: andere Szenarien testen, zB. 30-40 Mhz Bandbreite
    # Todo: Zielband 1785 - 1805 MHz --> >20 MHz 30.72 MHz = LTE
    # TODO: Frequenzantworten bei den "komischen" Ausreißern ansehen --> Untersuchen der Störung
    # TODO: cfo correction - Paper siehe Nick
    # TODO: T(t,f) in dB


    corr = correlate(data, zc_seq, mode = "full")
    corr_max = np.max(corr)

    recieved_power = []
    if len(data) % batchsize:
        print("There has been an over or underflow. Please repeat the measurement!") # maybe raise an exception
    number_of_batches = int( (len(data)) / batchsize )
    batches = []
    for b in range(0,number_of_batches):
        lower = batchsize * b
        upper = batchsize * (b+1)
        #calculate the recieved power
        recieved_power.append(np.sum(abs(data[lower:upper]**2)) / batchsize)
        #seperate the batches
        batches.append(corr[lower:upper])

    #create a time axis
    time = np.arange(0,capture_interval*number_of_batches, capture_interval)
    recieved_power_dbfs = 10 * np.log10(recieved_power)
    #plot the recieved power over time
    plt.figure(num="P(t)")
    plt.plot(time, recieved_power_dbfs)
    plt.xlabel("Time in [s]")
    plt.ylabel("Power of detected signal in [dBFS]")
    plt.title("Recieved signal power over time")
    #plt.show()

    h_meas = []
    b=0
    for batch in batches:
        batch_max = np.max(batch)
        xpeaks = find_peaks(abs(batch), 0.9 * batch_max, distance = 200)[0]
        h=[]
        for peak in xpeaks:
            peak_ =peak + b*batchsize
            h.append(corr[peak_-10:peak_+118])
        h_meas.append( np.mean(h[1:], axis =0))
        b+=1
    h_meas = np.array(h_meas)

    T = [] # timevariant transferfunction
    for h in h_meas:
        T.append(abs(np.fft.fftshift(np.fft.fft(h)))) # 128 point-fft
    T = np.array(T)
    #Cut Lowpass filter effects away
    T = T[:, 5:-5]
    cutoff_frequency = (1 - ( (1-len(T[0])) / 128 ) *fs)/2
    #create a frequency axis
    frequency = np.linspace( -cutoff_frequency, cutoff_frequency, len(T[0]))
    f, t = np.meshgrid(frequency, time)
    #plot transferfunction over time and frequency
    fig = plt.figure(num = "T(t,f)")
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("Time in [s]")
    ax.set_ylabel("Frequency in [Hz]")
    ax.set_zlabel("T(t,f)")
    ax.plot_surface(t, f, abs(T), cmap="plasma")



    #B_coh
    B_coh_50 = []
    B_coh_90 = []
    for t in range(len(T)):
        freq_corr_function = np.fft.fftshift( correlate(T[t],T[t], mode = "full") )
        delta_f = 2*(fs)/len(freq_corr_function)
        freq_corr_max = abs( np.max(freq_corr_function) )

        x1 = ( np.argmax( abs(freq_corr_function) < 0.5 * freq_corr_max ))
        B_coh_50.append( linear_interpolation_x(0.5 * freq_corr_max, abs(freq_corr_function[x1]), abs(freq_corr_function[x1 - 1]), x1, x1 - 1) * delta_f )

        x1 = ( np.argmax( abs(freq_corr_function) < 0.9 * freq_corr_max ))
        B_coh_90.append( linear_interpolation_x(0.9 * freq_corr_max, abs(freq_corr_function[x1]), abs(freq_corr_function[x1 - 1]), x1, x1 - 1) * delta_f )

    #Moving average filter
    windowsize_in_sec = 1
    batches_per_value = round(windowsize_in_sec/capture_interval)
    N = capture_interval * number_of_batches
    M = batches_per_value * capture_interval
    B_coh_50 = np.convolve(B_coh_50, np.ones(batches_per_value)/batches_per_value, mode = 'valid')
    B_coh_90 = np.convolve(B_coh_90, np.ones(batches_per_value)/batches_per_value, mode = 'valid')
    time = np.linspace(M/2, N-M/2, len(B_coh_50))

    plt.figure(num="B_coh(t)")
    plt.plot(time, B_coh_50, "-b", label = 'B_coh_50%(t)' )
    plt.plot(time, B_coh_90, "-r", label = 'B_coh_90%(t)' )
    plt.xlabel("Time in [s]")
    plt.ylabel("B_coh in [Hz]")
    plt.title("Coherence bandwidths over time with moving average filter applied")
    plt.ylim([0, fs])
    plt.legend(loc="best")

    B_coh_50_mean, B_coh_50_conf_interval_999 = confidence_interval(B_coh_50, 0.999)
    B_coh_90_mean, B_coh_90_conf_interval_999 = confidence_interval(B_coh_90, 0.999)
    print(f'99.9% of values of B_coh_50 lie in between {B_coh_50_conf_interval_999}[Hz]')
    print(f'99.9% of values of B_coh_90 lie in between {B_coh_90_conf_interval_999}[Hz]')

    #T_coh

    T = np.swapaxes(T,0,1)
    T_coh_50 = []
    T_coh_90 = []
    windowsize_in_sec = 1
    batches_per_value = round(windowsize_in_sec/capture_interval)
    step_width_in_batches = 1 if batches_per_value/10 < 1 else  round(batches_per_value/10)
    for t in range (0, int(number_of_batches-batches_per_value), step_width_in_batches):
        T_coh_50_f = []
        T_coh_90_f = []
        for f in range(len(T)):
            time_corr_function = np.fft.fftshift( correlate(T[f][t:t+batches_per_value],T[f][t:t+batches_per_value], mode = "full") )

            delta_t = 2*windowsize_in_sec/len(time_corr_function)
            time_corr_max = abs( np.max(time_corr_function) )

            x1 = ( np.argmax( abs(time_corr_function) < 0.5 * time_corr_max ))
            T_coh_50_f.append(linear_interpolation_x(0.5 * time_corr_max, abs(time_corr_function[x1]), abs(time_corr_function[x1 - 1]), x1, x1 - 1) * delta_t )

            x1 = ( np.argmax( abs(time_corr_function) < 0.9 * time_corr_max ))
            T_coh_90_f.append( linear_interpolation_x(0.9 * time_corr_max, abs(time_corr_function[x1]), abs(time_corr_function[x1 - 1]), x1, x1 - 1) * delta_t )

        T_coh_50.append(np.mean(T_coh_50_f))
        T_coh_90.append(np.mean(T_coh_90_f))

    T_coh_50_mean, T_coh_50_conf_interval_999 = confidence_interval(T_coh_50, 0.999)
    T_coh_90_mean, T_coh_90_conf_interval_999 = confidence_interval(T_coh_90, 0.999)
    print(f'99.9% of values of T_coh_50 lie in between {T_coh_50_conf_interval_999}[s]')
    print(f'99.9% of values of T_coh_90 lie in between {T_coh_90_conf_interval_999}[s]')

    time = np.linspace(windowsize_in_sec/2, number_of_batches * capture_interval-windowsize_in_sec/2, len(T_coh_50))
    plt.figure(num="T_coh(t)")
    plt.plot(time, T_coh_50, "-b", label = 'T_coh_50%(t)' )
    plt.plot(time, T_coh_90, "-r", label = 'T_coh_90%(t)' )
    plt.xlabel("Time in [s]")
    plt.ylabel("T_coh in [s]")
    plt.ylim([0, windowsize_in_sec])
    plt.title("Coherence times over time")
    plt.legend(loc="best")
    plt.show()
