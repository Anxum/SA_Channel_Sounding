import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, find_peaks
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D

fs = 0
fc = 0
batchsize = 0
capture_interval = 0
windowsize_in_sec = 1
batches_per_value = 0
number_of_batches = 0

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
    HIW = sigma * np.sqrt(var/n)
    return mean, np.array([mean - HIW, mean + HIW])


def read_meta_data(filename):
    str = filename.split('_')
    fc_ = int( str[3][0:str[3].find('MHz')] ) * 10**6
    fs_ = int( str[4][0:str[4].find('MSps')] ) * 10**6
    batchsize_ = int( str[5][0:str[5].find('S')] )
    capture_interval_ = int( str[6][0:str[6].find('ms')] ) * 10**-3
    print(f'Date of measurement = {str[1]}')
    print(f'Time of measurement = {str[2]}')
    print(f'fc = {fc_ * 10**-6} MHz, fs = {fs_ * 10**-6} MHz, batchsize = {batchsize_} S, capture interval = {capture_interval_ * 10**3} ms')
    return fc_, fs_, batchsize_, capture_interval_

def moving_average(data, windowsize, stepsize = 1):
    if stepsize < 1 or stepsize > windowsize:
        print("The stepsize should be between 1 and the windowsize")
    else:
        avg_data = []
        for n in range(0, len(data)-windowsize, stepsize):
            avg_data.append(np.mean(data[n:n + windowsize]))
        return np.array(avg_data)

def devide_in_batches(data, batchsize):
    if len(data) % batchsize:
        print("There has been an over or underflow. Please repeat the measurement!") # maybe raise an exception
    number_of_batches = int( (len(data)) / batchsize )
    batches = []
    for b in range(0, number_of_batches):
        lower = batchsize * b
        upper = batchsize * (b+1)
        #seperate the batches
        batches.append(data[lower:upper])
    return np.array(batches)

def plot_recieved_power(batches, time):
    recieved_power = []
    for batch in batches:
        recieved_power.append(np.sum(abs(batch**2)) / len(batch))
    recieved_power = np.array(recieved_power)
    #create a time axis
    recieved_power_dbfs = 10 * np.log10(recieved_power)
    #plot the recieved power over time
    plt.figure(num="P(t)")
    plt.plot(time, recieved_power_dbfs)
    plt.xlabel("Time in [s]")
    plt.ylabel("Power of detected signal in [dBFS]")
    plt.title("Recieved signal power over time")

def calculate_impulse_response(batches):
    corr = []
    for batch in batches:
        corr.append(correlate(batch, zc_seq, mode = "full"))
    corr = np.array(corr)
    h_mean = []
    for c in corr:
        c_max = np.max(abs(c))
        xpeaks = find_peaks(abs(c), 0.9 * c_max, distance = 200)[0]
        h=[]
        for peak in xpeaks:
            h.append(c[peak-10:peak+118])
        h = np.array(h)
        h_mean.append( np.mean(h, axis =0))
    return np.array(h_mean)

def plot_impulse_response(h, time, delay):
    tau,t = np.meshgrid(delay, time)
    fig = plt.figure(num = "h(t,delta_t)")
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("Time in [s]")
    ax.set_ylabel("Delay in [s]")
    ax.set_zlabel("h(t,delta_t)")
    ax.plot_surface(t, tau, abs(h), cmap="plasma")

def calculate_timevariant_transferfunction(h, unusable_bins = 5):

    T = [] # timevariant transferfunction
    for h_ in h:
        T.append(abs(np.fft.fftshift(np.fft.fft(h_)))) # 128 point-fft
    T = np.array(T)
    #Cut Lowpass filter effects away
    return T[:, unusable_bins:-unusable_bins]

def plot_timevariant_transferfunction(T, time, frequency):
    f, t = np.meshgrid(frequency, time)
    #plot transferfunction over time and frequency
    fig = plt.figure(num = "T(t,f)")
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("Time in [s]")
    ax.set_ylabel("Frequency in [Hz]")
    ax.set_zlabel("T(t,f)")
    ax.plot_surface(t, f, abs(T), cmap="plasma")

def calculate_B_coh(T, threshold, stepsize = 10):
    if not 0<=threshold<=1:
        print("threshold value has to be between 0 and 1")
    B_coh = []
    for t in T:
        freq_corr_function = np.fft.fftshift( correlate(t,t, mode = "full") )
        delta_f = 2*(fs)/len(freq_corr_function)
        freq_corr_max = abs( np.max(freq_corr_function) )

        x1 = ( np.argmax( abs(freq_corr_function) < threshold * freq_corr_max ))
        B_coh.append( linear_interpolation_x(threshold * freq_corr_max, abs(freq_corr_function[x1]), abs(freq_corr_function[x1 - 1]), x1, x1 - 1) * delta_f )
    B_coh = np.array(B_coh)
    return moving_average(B_coh, batches_per_value, stepsize)

def cfo_correction(function, t_axis, upsampled_axis):

    T_ = t_axis[1]-t_axis[0]
    upsampled_function = np.zeros(np.shape(upsampled_axis))
    for n in range(len(function)):
        upsampled_function = np.add(upsampled_function, function[n]*np.sinc(upsampled_axis/T_-n))
    np.array(upsampled_function)
    max1 = np.argmax(abs(function))
    max2 = np.argmax(abs(upsampled_function))
    offset =  max2 - max1 * upsample_factor
    print(offset)
    negative_offset = offset < 0
    if negative_offset :
        offset = offset + upsample_factor

    new_h = np.array( upsampled_function[ offset :: upsample_factor])
    if negative_offset:
        new_h = np.roll(new_h, 1)
        new_h[0] = 0
    return new_h

def plot_B_coh(b_50, b_90, time):

    plt.figure(num="B_coh(t)")
    plt.plot(time, b_50, "-b", label = 'B_coh_50%(t)' )
    plt.plot(time, b_90, "-r", label = 'B_coh_90%(t)' )
    plt.xlabel("Time in [s]")
    plt.ylabel("B_coh in [Hz]")
    plt.title("Coherence bandwidths over time with moving average filter applied")
    plt.ylim([0, fs])
    plt.legend(loc="best")

def calculate_T_coh(T_, threshold):
    T = np.swapaxes(T_,0,1)
    T_coh = []
    stepsize = 1 if batches_per_value/10 < 1 else  round(batches_per_value/10)
    for t in range (0, int(number_of_batches-batches_per_value), stepsize):
        T_coh_f = []
        for f in T:
            time_corr_function = np.fft.fftshift( correlate(f[t:t+batches_per_value],f[t:t+batches_per_value], mode = "full") )

            delta_t = 2*windowsize_in_sec/len(time_corr_function)
            time_corr_max = abs( np.max(time_corr_function) )

            x1 = ( np.argmax( abs(time_corr_function) < threshold * time_corr_max ))
            T_coh_f.append(linear_interpolation_x(threshold * time_corr_max, abs(time_corr_function[x1]), abs(time_corr_function[x1 - 1]), x1, x1 - 1) * delta_t )

        T_coh.append(np.mean(T_coh_f))
    return np.array(T_coh)

def plot_T_coh(b_50, b_90, time):
    plt.figure(num="T_coh(t)")
    plt.plot(time, T_coh_50, "-b", label = 'T_coh_50%(t)' )
    plt.plot(time, T_coh_90, "-r", label = 'T_coh_90%(t)' )
    plt.xlabel("Time in [s]")
    plt.ylabel("T_coh in [s]")
    plt.ylim([0, windowsize_in_sec])
    plt.title("Coherence times over time")
    plt.legend(loc="best")



if __name__ == "__main__":

        # Todo: andere Szenarien testen, zB. 30-40 Mhz Bandbreite
        # Todo: Zielband 1785 - 1805 MHz --> >20 MHz 30.72 MHz = LTE
        # TODO: Frequenzantworten bei den "komischen" Ausreißern ansehen --> Untersuchen der Störung
        # TODO: cfo correction - Paper siehe Nick
        # TODO: T(t,f) in dB

    fc, fs, batchsize, capture_interval = read_meta_data(file)
    batches_per_value = round(windowsize_in_sec/capture_interval)
    data = np.fromfile(open(f"{folder}/{file}"), dtype=np.complex64)
    zc_seq = np.load('zc_sequence.npy')

    batches = devide_in_batches(data, batchsize)
    number_of_batches = len(batches)
    time = np.arange(0,capture_interval*len(batches), capture_interval)
    plot_recieved_power(batches, time)

    h = calculate_impulse_response(batches)
    np.save("impulse_response", h)
    delay = np.linspace(0, 128 / fs, np.shape(h)[1])
    plot_impulse_response(h, time, delay)

    T = calculate_timevariant_transferfunction(h)
    cutoff_frequency = (1 - ( (1-np.shape(T)[1]) / 128 ) *fs)/2
    frequency = np.linspace( -cutoff_frequency, cutoff_frequency, np.shape(T)[1])
    plot_timevariant_transferfunction(T, time, frequency)

    B_coh_50 = calculate_B_coh(T, 0.5)
    B_coh_90 = calculate_B_coh(T, 0.9)

    signal_time = capture_interval * len(batches)
    time_of_movavg_filter = batches_per_value * capture_interval
    time = np.linspace(time_of_movavg_filter/2, signal_time-time_of_movavg_filter/2, len(B_coh_50))

    plot_B_coh(B_coh_50, B_coh_90, time)

    B_coh_50_mean, B_coh_50_conf_interval_999 = confidence_interval(B_coh_50, 0.999)
    B_coh_90_mean, B_coh_90_conf_interval_999 = confidence_interval(B_coh_90, 0.999)
    print(f'99.9% of values of B_coh_50 lie in between {B_coh_50_conf_interval_999}[Hz]')
    print(f'99.9% of values of B_coh_90 lie in between {B_coh_90_conf_interval_999}[Hz]')

    #T_coh
    windowsize_in_sec = 1
    batches_per_value = round(windowsize_in_sec/capture_interval)
    T_coh_50 = calculate_T_coh(T, 0.5)
    T_coh_90 = calculate_T_coh(T, 0.9)

    time = np.linspace(windowsize_in_sec/2, signal_time - windowsize_in_sec/2, len(T_coh_50))

    plot_T_coh(T_coh_50, T_coh_90, time)

    T_coh_50_mean, T_coh_50_conf_interval_999 = confidence_interval(T_coh_50, 0.999)
    T_coh_90_mean, T_coh_90_conf_interval_999 = confidence_interval(T_coh_90, 0.999)
    print(f'99.9% of values of T_coh_50 lie in between {T_coh_50_conf_interval_999}[s]')
    print(f'99.9% of values of T_coh_90 lie in between {T_coh_90_conf_interval_999}[s]')

    plt.show()
