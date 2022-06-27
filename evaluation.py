import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, resample
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D

fs = 0
fc = 0
batchsize = 0
capture_interval = 0
windowsize_in_sec = 1
batches_per_value = 0
number_of_batches = 0

folder = "../Messungen/Testmessungen/1795_MHz_10MHz_TX_movement"
file = "capture_2022-06-14_17-45-49_1795MHz_10MSps_1024S_10ms.dat"
# TODO UMTS Antenne Patchantenne / Chipantenne recherchieren

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

def cyclic_correlate(a, v):
    arr = np.tile(np.fft.fftshift(a), 2)
    return np.correlate(arr, v, mode = "valid")

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
    for b in range(number_of_batches):
        lower = batchsize * b
        upper = batchsize * (b+1)
        #seperate the batches
        batches.append(data[lower:upper])
    return np.array(batches)

def plot_recieved_power(batches, time):
    recieved_power = []
    for batch in batches:
        recieved_power.append(np.sum(abs(batch**2)) / batchsize)
    recieved_power = np.array(recieved_power)
    recieved_power_dbfs = 10 * np.log10(recieved_power)
    #plot the recieved power over time
    plt.figure(num="P(t)")
    plt.plot(time, recieved_power_dbfs)
    plt.xlabel("Time [s]")
    plt.ylabel("Power of detected signal [dBFS]")
    plt.ylim([-80, 0])
    plt.title("Received signal power over time")

def calculate_impulse_response(batches):
    corr = batches
    for idx,c in enumerate(corr.copy()):
        c = np.correlate(c, zc_seq, mode = "same")/batchsize
        c = cfo_correction(c, 10)
        corr[idx] = c
    corr = np.reshape(corr, (number_of_batches, int(round(batchsize/256)), 256))
    corr = corr[:,1:-1,:]
    for idx_c, c in enumerate(corr.copy()):
        for idx_h, h in enumerate(c.copy()):
            h_max = np.argmax(abs(h))
            h = np.roll(h,10-h_max)
            corr[idx_c,idx_h] = h
    corr = corr[:,:,:128]
    return np.mean(corr, axis = 1)

def plot_impulse_response(h, time, delay):
    tau,t = np.meshgrid(delay, time)
    fig = plt.figure(num = "h(t,delta_t)")
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Delay [s]")
    ax.set_zlabel("h(t,delta_t)")
    ax.plot_surface(t, tau, abs(h), cmap="plasma")

def calculate_timevariant_transferfunction(h, usable_bandwidth = 20e6):
    T = [] # timevariant transferfunction
    for h_ in h:
        T.append(abs(np.fft.fftshift(np.fft.fft(h_, norm = "ortho")))) # 128 point-fft
    T = np.array(T)
    #Cut Lowpass filter effects away to 20 MHz
    if usable_bandwidth < fs:
        unusable_bins = round(64 * (1-usable_bandwidth/fs))
        T = T[:, unusable_bins:-unusable_bins]
    return T

def plot_timevariant_transferfunction(T, time, frequency):
    f, t = np.meshgrid(frequency, time)
    #plot transferfunction over time and frequency
    fig = plt.figure(num = "T(t,f) [dBFS]")
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_zlabel("T(t,f) [dBFS]")
    ax.plot_surface(t, f, 10*np.log10(abs(T)), cmap="plasma")

def calculate_B_coh(T, threshold, stepsize = 10):
    if not 0<=threshold<=1:
        print("threshold value has to be between 0 and 1")
    B_coh = []
    for t in T:
        freq_corr_function = np.fft.fftshift( cyclic_correlate(t,t) )
        freq_corr_max = abs( np.max(freq_corr_function) )

        x = ( np.argmax( abs(freq_corr_function) < threshold * freq_corr_max ))
        if np.min(abs(freq_corr_function) > threshold * freq_corr_max ):
            B_coh.append(fs)
        else:
            B_coh.append(x * 2*(fs)/len(freq_corr_function))

        #B_coh.append( linear_interpolation_x(threshold * freq_corr_max, abs(freq_corr_function[x1]), abs(freq_corr_function[x1 - 1]), x1, x1 - 1) * delta_f )
    B_coh = np.array(B_coh)
    return moving_average(B_coh, batches_per_value, stepsize)

# def sinc_interp(x_len, upsample_factor):
#     '''
#     Modified sinc interpolation from this source: https://gist.github.com/endolith/1297227#file-sinc_interp-py
#     '''
#     s = np.arange(0, x_len* upsample_factor, upsample_factor)
#     u = np.arange(x_len* upsample_factor, 2 * x_len* upsample_factor)
#     sinc_idx = np.tile(u, (x_len, 1)) - np.tile(s[:,np.newaxis], (1, len(u)))
#     u_sinc = np.arange(-x_len, x_len, 1/(upsample_factor))
#     sinc = np.sinc(u_sinc)
#     sincM = sinc[sinc_idx]
#     print(np.shape(sincM))
#     plt.figure()
#     plt.plot(sinc)
#     plt.figure()
#     plt.plot(10*np.log10(abs(sinc)))
#     plt.show()
#     return sincM

def cfo_correction(x, upsample_factor):
    # upsample_factor = np.shape(sincM)[1] / np.shape(sincM)[0]
    # upsampled_function = np.dot(x,sincM)
    upsampled_function = resample(x, len(x) * upsample_factor)
    f_max = np.max(abs(upsampled_function))
    x_max = np.max(abs(x))
    x1peaks = find_peaks(abs(upsampled_function), 0.9 * f_max, distance = 200 * upsample_factor)[0]
    x2peaks = find_peaks(abs(x), 0.9 * x_max, distance = 200)[0]

    corrected_diff = np.mean(np.diff(x1peaks)) / 256
    offset =  round(np.mean(x1peaks % corrected_diff)) % upsample_factor
    resampled_axis = np.rint(np.arange(offset, len(upsampled_function)-1, corrected_diff)).astype(int)
    #print(np.shape(resampled_axis))
    new_h = upsampled_function[resampled_axis]
    if len(new_h) >len(x):
        new_h = new_h[:len(x)]
    if len(new_h) < len(x):
        new_h = np.pad(new_h, (0,len(x) - len(new_h)), 'constant')
    # plt.figure()
    # plt.plot(abs(upsampled_function))
    # plt.figure()
    # plt.plot(abs(new_h))
    # plt.plot(abs(x))
    # plt.show()
    # if np.shape(new_h)[0] != 128:
    #      print(f'len(new_h) = {np.shape(new_h)[0]}')
    #      print(f'len(upsampled_function) = {np.shape(upsampled_function)[0]}')
    #      print(f'upsample_factor = {upsample_factor}')
    #      print(f'offset = {offset}')
    # new_max = np.argmax(abs(new_h))
    # if new_max != max1:
    #     diff = max1 - new_max
    #     new_h = np.roll(new_h, diff)
    #     if diff>0:
    #         new_h[:diff] = 0
    #     else:
    #         new_h[diff:] = 0
    # new_max = np.argmax(abs(new_h))
    # if new_h[new_max] != upsampled_function[max2]:
    #     print(new_h[new_max])
    #     print(upsampled_function[max2])
    # if new_max != max1:
    #     print(new_max, max1)
    #     plt.figure()
    #     plt.plot(abs(new_h))
    #     plt.plot(abs(function))
    #     plt.show()
    return new_h

def plot_B_coh(b_50, b_90, time):

    plt.figure(num="B_coh(t)")
    plt.plot(time, b_50, "-b", label = 'B_coh_50%(t)' )
    plt.plot(time, b_90, "-r", label = 'B_coh_90%(t)' )
    plt.xlabel("Time [s]")
    plt.ylabel("B_coh [Hz]")
    plt.title("Coherence bandwidths over time with moving average filter applied")
    plt.ylim([0
    , 1.02*fs])
    plt.legend(loc="best")

def calculate_T_coh(T_, threshold):
    T = np.swapaxes(T_,0,1)
    T_coh = []
    stepsize = 1 if batches_per_value/10 < 1 else  round(batches_per_value/10)
    for t in range (0, int(number_of_batches-batches_per_value), stepsize):
        T_coh_f = []
        for f in T:
            time_corr_function = np.fft.fftshift( cyclic_correlate(f[t:t+batches_per_value],f[t:t+batches_per_value]) )
            time_corr_max = abs( np.max(time_corr_function) )

            x = ( np.argmax( abs(time_corr_function) < threshold * time_corr_max ))
            if np.min(abs(time_corr_function) > threshold * time_corr_max ):
                T_coh_f.append(windowsize_in_sec)
            else:
                T_coh_f.append(x * 2*windowsize_in_sec/len(time_corr_function))
            #T_coh_f.append(linear_interpolation_x(threshold * time_corr_max, abs(time_corr_function[x1]), abs(time_corr_function[x1 - 1]), x1, x1 - 1) * delta_t )

        T_coh.append(np.mean(T_coh_f))
    return np.array(T_coh)

def plot_T_coh(b_50, b_90, time):
    plt.figure(num="T_coh(t)")
    plt.plot(time, T_coh_50, "-b", label = 'T_coh_50%(t)' )
    plt.plot(time, T_coh_90, "-r", label = 'T_coh_90%(t)' )
    plt.xlabel("Time [s]")
    plt.ylabel("T_coh [s]")
    plt.ylim([0, 1.02 * windowsize_in_sec])
    plt.title("Coherence times over time")
    plt.legend(loc="best")



if __name__ == "__main__":

        # TODO: Szenarien mit 30, validieren mit synthetischen Daten(bekannte Kohärenz BB, Kohärenzzseit Siehe Matlab)

    fc, fs, batchsize, capture_interval = read_meta_data(file)
    batches_per_value = round(windowsize_in_sec/capture_interval)
    data = np.fromfile(open(f"{folder}/{file}"), dtype=np.complex64)
    zc_seq = np.load('zc_sequence.npy')

    batches = devide_in_batches(data, batchsize)
    number_of_batches = len(batches)
    time = np.arange(0,capture_interval*len(batches), capture_interval)
    plot_recieved_power(batches, time)

    delay = np.linspace(0, 128 / fs, 128)
    h = calculate_impulse_response(batches)

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
