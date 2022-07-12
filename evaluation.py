import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from file_management import register_measurements, choose_measurement, read_meta_data, save_impulse_response
from graphing import plot_recieved_power, plot_impulse_response, plot_timevariant_transferfunction, plot_B_coh, plot_T_coh
from scipy.signal import find_peaks, resample
from scipy.stats import norm, moment
from mpl_toolkits.mplot3d import Axes3D


fs = 0
fc = 0
batchsize = 0
capture_interval = 0
windowsize_in_sec = 1
batches_per_value = 0
number_of_batches = 0
zc_len = 0


def confidence_interval(data, percentage):
    quantile = (percentage + 1)/2
    mean = np.mean(data)
    var = np.var(data)
    n = len(data)
    sigma = norm.ppf(quantile)
    HIW = sigma * np.sqrt(var/n)
    return mean, np.array([mean - HIW, mean + HIW])

def cyclic_auto_correlate(a):
    arr = np.tile(np.fft.fftshift(a), 2)
    return np.correlate(arr, a, mode = "valid")


def moving_average(data, windowsize, stepsize = 1):
    if stepsize < 1 or stepsize > windowsize:
        print("When calculating a moving average: The stepsize should be between 1 and the windowsize")
    else:
        avg_data = []
        for n in range(0, len(data)-windowsize, stepsize):
            avg_data.append(np.mean(data[n:n + windowsize]))
        return np.array(avg_data)

def devide_in_batches(data, batchsize):
    if len(data) % batchsize:
        print("There has been an over or underflow. Please repeat the measurement!") # maybe raise an exception
    number_of_batches = int( (len(data)) / batchsize )
    return np.reshape(data, (number_of_batches, batchsize))


def correlate_batches(batches):
    corr = batches
    for idx,c in enumerate(corr.copy()):
        c = np.correlate(c, zc_seq, mode = "same")/batchsize
        if len(c) == 0:
            return []
        corr[idx] = c
    corr = np.reshape(corr, (number_of_batches, int(round(batchsize/zc_len)), zc_len))
    for idx_c, c in enumerate(corr.copy()):
        for idx_h, h in enumerate(c.copy()):
            h_max = np.argmax(abs(h))
            h = np.roll(h,10-h_max)
            corr[idx_c,idx_h] = h
    return corr


def calculate_timevariant_transferfunction(h, usable_bandwidth = 20e6):
    T = [] # timevariant transferfunction
    for h_ in h:
        T.append(abs(np.fft.fftshift(np.fft.fft(h_, norm = "ortho")))) # 128 point-fft
    T = np.array(T)
    #Cut Lowpass filter effects away to 20 MHz
    if usable_bandwidth < fs:
        unusable_bins = round(np.shape(h)[1]/2 * (1-usable_bandwidth/fs))
        T = T[:, unusable_bins:-unusable_bins]
    return T

def cfo_correction(corr, upsample_factor):
    batchsize = int(np.shape(corr)[1] * np.shape(corr)[2])
    corr_ = np.reshape(corr, (np.shape(corr)[0], batchsize))
    corrected_corr = np.zeros(np.shape(corr_), dtype =np.complex64)
    for idx, x in enumerate(corr_.copy()):

        upsampled_function = resample(x, len(x) * upsample_factor)
        h_max = []
        for offset in range(upsample_factor):
            resampled_axis = np.rint(np.arange(offset, len(upsampled_function)-1, upsample_factor)).astype(int)
            h_max.append(np.max(abs(upsampled_function[resampled_axis])))
        offset = np.argmax(h_max)
        resampled_axis = np.rint(np.arange(offset, len(upsampled_function)-1, upsample_factor)).astype(int)
        new_h = upsampled_function[resampled_axis]
        if len(new_h) >len(x):
            new_h = new_h[:len(x)]
        if len(new_h) < len(x):
            new_h = np.pad(new_h, (0,len(x) - len(new_h)), 'constant')
        corrected_corr[idx] = new_h
    return np.reshape(corrected_corr, np.shape(corr))

def calculate_impulse_response(corr):
    corr = corr[:,1:-1,:]
    return np.mean(corr, axis = 1)

def calculate_B_coh_cyclic_correlate(T, threshold, stepsize = 10):
    if not 0<=threshold<=1:
        print("When calculating B_coh: threshold value has to be between 0 and 1")
    B_coh = []
    for t in T:
        freq_corr_function = np.fft.fftshift( cyclic_auto_correlate(t) )
        freq_corr_max = abs( np.max(freq_corr_function) )

        x = ( np.argmax( abs(freq_corr_function) < threshold * freq_corr_max ))
        if np.min(abs(freq_corr_function)) > threshold * freq_corr_max :
            B_coh.append(np.min([fs, 20e6]))
        else:
            B_coh.append(x * 2*(np.min([fs, 20e6]))/len(freq_corr_function))

    B_coh = np.array(B_coh)
    return moving_average(B_coh, batches_per_value, stepsize)


def calculate_T_coh_cyclic_correlate(T_, threshold):
    T = np.swapaxes(T_,0,1)
    T_coh = []
    stepsize = 1 if batches_per_value/10 < 1 else  round(batches_per_value/10)
    for t in range (0, int(number_of_batches-batches_per_value), stepsize):
        T_coh_f = []
        for f in T:
            time_corr_function = np.fft.fftshift( cyclic_auto_correlate(f[t:t+batches_per_value]) )

            time_corr_max = abs( np.max(time_corr_function) )

            x = ( np.argmax( abs(time_corr_function) < threshold * time_corr_max ))
            if np.min(abs(time_corr_function)) > threshold * time_corr_max :
                T_coh_f.append(windowsize_in_sec)
            else:
                T_coh_f.append(x * 2*windowsize_in_sec/len(time_corr_function))
        T_coh.append(np.mean(T_coh_f))

    return np.array(T_coh)

def calculate_B_coh_auto_correlate(T, threshold, stepsize = 10):
    if not 0<=threshold<=1:
        print("When calculating B_coh: threshold value has to be between 0 and 1")
    B_coh = []
    for t in T:
        freq_corr_function = np.fft.fftshift(np.correlate(t,t,mode = "full") )
        freq_corr_max = abs( np.max(freq_corr_function) )

        x = ( np.argmax( abs(freq_corr_function) < threshold * freq_corr_max ))
        if np.min(abs(freq_corr_function)) > threshold * freq_corr_max :
            B_coh.append(np.min([fs, 20e6]))
        else:
            B_coh.append(x * 2*(np.min([fs, 20e6]))/len(freq_corr_function))

    B_coh = np.array(B_coh)
    return moving_average(B_coh, batches_per_value, stepsize)

def calculate_T_coh_auto_correlate(T_, threshold):
    T = np.swapaxes(T_,0,1)
    T_coh = []
    stepsize = 1 if batches_per_value/10 < 1 else  round(batches_per_value/10)
    for t in range (0, int(number_of_batches-batches_per_value), stepsize):
        T_coh_f = []
        for f in T:
            time_corr_function = np.fft.fftshift(np.correlate(f[t:t+batches_per_value],f[t:t+batches_per_value], mode = "full") )

            time_corr_max = abs( np.max(time_corr_function) )

            x = ( np.argmax( abs(time_corr_function) < threshold * time_corr_max ))
            if np.min(abs(time_corr_function)) > threshold * time_corr_max :
                T_coh_f.append(windowsize_in_sec)
            else:
                T_coh_f.append(x * 2*windowsize_in_sec/len(time_corr_function))
        T_coh.append(np.mean(T_coh_f))

    return np.array(T_coh)

def calculate_B_coh_sliding_window():
    pass
def calculate_T_coh_sliding_window(T_, threshold):
    T = np.swapaxes(T_,0,1)
    T_coh = []
    stepsize = 1 if batches_per_value/10 < 1 else  round(batches_per_value/10)
    for f in T:
        T_coh_f = []
        for t in range (0, int(number_of_batches-batches_per_value), stepsize):
            window = f[t:t+batches_per_value]
            function_window = np.roll(f, -t+batches_per_value)[0:3*batches_per_value]
            time_corr_function = np.fft.fftshift(np.correlate(function_window,window, mode = "full"))
            #plt.figure()
            #plt.plot(window)
            #plt.plot(function_window)
            #plt.plot(5*time_corr_function)
            #plt.show()
            time_corr_max = time_corr_function[0]
            x = ( np.argmax( abs(time_corr_function) < threshold * time_corr_max ))
            T_coh_f.append(min(x * 2*windowsize_in_sec/len(time_corr_function), windowsize_in_sec))
        T_coh.append(np.mean(T_coh_f))

    return np.array(T_coh)


def calculate_B_coh_power_delay_spread(h, threshold):
    Td = []
    for h_t in h:
        Td.append(moment(h_t, moment = 2))
    Td = np.array(Td)
    if threshold == 0.5:
        return 0.2/Td
    if threshold == 0.9:
        return 0.02/Td




if __name__ == "__main__":

    # TODO: validieren mit synthetischen Daten(bekannte Kohärenz BB, Kohärenzzseit Siehe Matlab)

    register_measurements()
    while True:
        folder, file, name_of_measurement = choose_measurement()
        print("===========================================================================")
        if folder == "" and file == "" and name_of_measurement == "":
            break
        fc, fs, batchsize, capture_interval, date_of_measurement, time_of_measurement = read_meta_data(file)
        print("")
        print(f'Calculating measurement with the name: {name_of_measurement}')
        print(f'Date of measurement = {date_of_measurement}')
        print(f'Time of measurement = {time_of_measurement}')
        print(f'fc = {fc * 1e-6} MHz, fs = {fs * 1e-6} MHz, batchsize = {batchsize} S, capture interval = {capture_interval * 1e3} ms')
        print("")
        h = 0
        recieved_power_dbfs =0
        batches_per_value = round(windowsize_in_sec/capture_interval)
        if ".dat"in file:
            print("Reading from dat file")
            data = np.fromfile(open(f"{folder}/{file}"), dtype=np.complex64)
            zc_seq = np.load('zc_sequence.npy')
            zc_len = len(zc_seq)

            batches = devide_in_batches(data, batchsize)
            number_of_batches = np.shape(batches)[0]
            time = np.arange(0,capture_interval*batchsize, capture_interval)

            recieved_power = np.sum(abs(batches**2), axis = 1) / batchsize
            recieved_power_dbfs = 10 * np.log10(recieved_power)

            corr = correlate_batches(batches)
            corr_cfo_corrected = cfo_correction(corr, 10)
            h = calculate_impulse_response(corr_cfo_corrected)
            save_impulse_response(h,recieved_power_dbfs, date_of_measurement, time_of_measurement, fc, fs, batchsize, capture_interval, name_of_measurement, f'{folder}/{file}')
        if ".npz" in file:
            print("Reading from npz file")
            meas = np.load(f'{folder}/{file}',allow_pickle=True)
            h = meas["name1"]
            recieved_power_dbfs = meas["name2"]
            number_of_batches = len(recieved_power_dbfs)
        time = np.arange(0,capture_interval*len(recieved_power_dbfs), capture_interval)
        delay = np.linspace(0, np.shape(h)[1] / fs, np.shape(h)[1])
        if np.max(recieved_power_dbfs) > -10:
            print("Warning: The recieved signal power exceeds -10 dBFS. Clipping may occour!")
            print("")
        plot_recieved_power(recieved_power_dbfs, time)
        plot_impulse_response(h, time, delay)


        T = calculate_timevariant_transferfunction(h)
        cutoff_frequency = (1 - ( (1-np.shape(T)[1]) / np.shape(h)[1] ) *fs)/2
        frequency = np.linspace( -cutoff_frequency, cutoff_frequency, np.shape(T)[1])
        plot_timevariant_transferfunction(T, time, frequency)

        B_coh_50 = calculate_B_coh_power_delay_spread(T, 0.5)
        B_coh_90 = calculate_B_coh_power_delay_spread(T, 0.9)

        signal_time = capture_interval * number_of_batches
        time_of_movavg_filter = batches_per_value * capture_interval
        time = np.linspace(time_of_movavg_filter/2, signal_time-time_of_movavg_filter/2, len(B_coh_50))

        plot_B_coh(B_coh_50, B_coh_90, time, np.min([fs,20e6]))

        B_coh_50_mean, B_coh_50_conf_interval_999 = confidence_interval(B_coh_50, 0.999)
        B_coh_90_mean, B_coh_90_conf_interval_999 = confidence_interval(B_coh_90, 0.999)
        print(f'99.9% of values of B_coh_50 lie in between {B_coh_50_conf_interval_999}[Hz]')
        print(f'99.9% of values of B_coh_90 lie in between {B_coh_90_conf_interval_999}[Hz]')

        #T_coh
        windowsize_in_sec = 1
        batches_per_value = round(windowsize_in_sec/capture_interval)
        T_coh_50 = calculate_T_coh_sliding_window(T, 0.5)
        T_coh_90 = calculate_T_coh_sliding_window(T, 0.9)

        time = np.linspace(windowsize_in_sec/2, signal_time - windowsize_in_sec/2, len(T_coh_50))

        plot_T_coh(T_coh_50, T_coh_90, time, windowsize_in_sec)

        T_coh_50_mean, T_coh_50_conf_interval_999 = confidence_interval(T_coh_50, 0.999)
        T_coh_90_mean, T_coh_90_conf_interval_999 = confidence_interval(T_coh_90, 0.999)
        print(f'99.9% of values of T_coh_50 lie in between {T_coh_50_conf_interval_999}[s]')
        print(f'99.9% of values of T_coh_90 lie in between {T_coh_90_conf_interval_999}[s]')
        print("")
        plt.show()
