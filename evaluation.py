import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from file_management import register_measurements, choose_measurement, read_meta_data, save_impulse_response
from graphing import plot_recieved_power, plot_impulse_response, plot_timevariant_transferfunction, plot_B_coh, plot_T_coh
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

# TODO UMTS Antenne Patchantenne / Chipantenne recherchieren

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


def correlate_batches(batches):
    corr = batches
    for idx,c in enumerate(corr.copy()):
        c = np.correlate(c, zc_seq, mode = "same")/batchsize
        if len(c) == 0:
            return []
        corr[idx] = c
    corr = np.reshape(corr, (number_of_batches, int(round(batchsize/256)), 256))
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
        unusable_bins = round(64 * (1-usable_bandwidth/fs))
        T = T[:, unusable_bins:-unusable_bins]
    return T

def cfo_correction(corr, upsample_factor):
    batchsize = int(np.shape(corr)[1] * np.shape(corr)[2])
    corr_ = np.reshape(corr, (np.shape(corr)[0], batchsize))
    corrected_corr = np.zeros(np.shape(corr_), dtype =np.complex64)
    for idx, x in enumerate(corr_.copy()):

        upsampled_function = resample(x, len(x) * upsample_factor)
        f_max = np.max(abs(upsampled_function))
        x_max = np.max(abs(x))
        x1peaks = find_peaks(abs(upsampled_function), 0.9 * f_max, distance = 200 * upsample_factor)[0]
        x2peaks = find_peaks(abs(x), 0.9 * x_max, distance = 200)[0]

        if len(x1peaks) < batchsize / (2 * 256) or len(x1peaks) > 2*batchsize / 256:
            print("Bad signal! Could not perform cfo correction!")
            print("")
            return corr

        corrected_diff = np.mean(np.diff(x1peaks)) / 256
        offset =  round(np.mean(x1peaks % corrected_diff)) % upsample_factor
        resampled_axis = np.rint(np.arange(offset, len(upsampled_function)-1, corrected_diff)).astype(int)
        new_h = upsampled_function[resampled_axis]
        if len(new_h) >len(x):
            new_h = new_h[:len(x)]
        if len(new_h) < len(x):
            new_h = np.pad(new_h, (0,len(x) - len(new_h)), 'constant')
        corrected_corr[idx] = new_h
    return np.reshape(corrected_corr, np.shape(corr))

def calculate_impulse_response(corr):
    corr = corr[:,1:-1,:128]
    return np.mean(corr, axis = 1)

def calculate_B_coh(T, threshold, stepsize = 10):
    if not 0<=threshold<=1:
        print("threshold value has to be between 0 and 1")
    B_coh = []
    for t in T:
        freq_corr_function = np.fft.fftshift( cyclic_correlate(t,t) )
        freq_corr_max = abs( np.max(freq_corr_function) )

        x = ( np.argmax( abs(freq_corr_function) < threshold * freq_corr_max ))
        if np.min(abs(freq_corr_function) > threshold * freq_corr_max ):
            B_coh.append(np.min([fs, 20e6]))
        else:
            B_coh.append(x * 2*(np.min([fs, 20e6]))/len(freq_corr_function))

    B_coh = np.array(B_coh)
    return moving_average(B_coh, batches_per_value, stepsize)


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

        T_coh.append(np.mean(T_coh_f))
    return np.array(T_coh)



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
        if ".dat"in file:
            batches_per_value = round(windowsize_in_sec/capture_interval)
            data = np.fromfile(open(f"{folder}/{file}"), dtype=np.complex64)
            zc_seq = np.load('zc_sequence.npy')

            batches = devide_in_batches(data, batchsize)
            number_of_batches = np.shape(batches)[0]
            time = np.arange(0,capture_interval*len(batches), capture_interval)

            recieved_power = np.sum(abs(batches**2), axis = 1) / batchsize
            recieved_power_dbfs = 10 * np.log10(recieved_power)
            if np.max(recieved_power_dbfs) > -10:
                print("Warning: The recieved signal power exceeds -10 dBFS. Clipping may occour!")
                print("")
            plot_recieved_power(recieved_power_dbfs, time)


            corr = correlate_batches(batches)
            corr_cfo_corrected = cfo_correction(corr, 10)
            h = calculate_impulse_response(corr_cfo_corrected)
            save_impulse_response(h, date_of_measurement, time_of_measurement, fc, fs, batchsize, capture_interval, name_of_measurement, f'{folder}/{file}')
        if ".npy" in file:
            h = np.load(f'{folder}/{file}')
        time = np.arange(0,capture_interval*batchsize, capture_interval)
        delay = np.linspace(0, 128 / fs, 128)
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

        plot_B_coh(B_coh_50, B_coh_90, time, np.min([fs,20e6]))

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

        plot_T_coh(T_coh_50, T_coh_90, time, windowsize_in_sec)

        T_coh_50_mean, T_coh_50_conf_interval_999 = confidence_interval(T_coh_50, 0.999)
        T_coh_90_mean, T_coh_90_conf_interval_999 = confidence_interval(T_coh_90, 0.999)
        print(f'99.9% of values of T_coh_50 lie in between {T_coh_50_conf_interval_999}[s]')
        print(f'99.9% of values of T_coh_90 lie in between {T_coh_90_conf_interval_999}[s]')
        print("")
        plt.show()
