import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from file_management import register_measurements, choose_measurement, read_meta_data, save_impulse_response
from graphing import plot_recieved_power, plot_impulse_response, plot_timevariant_transferfunction, plot_B_coh, plot_T_coh
from scipy.signal import resample
from scipy.stats import norm, moment


fs = 0
fc = 0
batchsize = 0
capture_interval = 0
windowsize_in_sec = 1
batches_per_value = 0
number_of_batches = 0
zc_len = 0


def confidence_interval(data, percentage):
    '''
    descrpition:
        calculate the confidence interval of an array. It i expected, that N is big enough
        to use the gauss distribution

    inputs:
        data             -  The array of datapoints
        percentage       -  The confidence value of the confidence interval

    returns:
        mean             -  The average value of data
        interval         -  The lower and upper bounds of the confidence interval
    '''
    quantile = (percentage + 1)/2
    mean = np.mean(data)
    var = np.var(data)
    n = len(data)
    sigma = norm.ppf(quantile)
    HIW = sigma * np.sqrt(var/n)
    return mean, np.array([mean - HIW, mean + HIW])

def cyclic_auto_correlate(a):
    '''
    description:
        calculate the cyclic auto correlation funtion. The length of the output will be the same as the length of the input

    inputs:
        a                -  The function to be correlated

    returns:
        corr_function    -  The cyclically calculated auto correlation function
    '''
    # build a funtion, a will be correlated with, by splitting the datapoints of a in half (start an end)
    # and rearranging them in the following order: end-start-end-start. This way, a can be shifted
    # half of it's length to the left and to the right.
    arr = np.tile(np.fft.fftshift(a), 2)
    # correlate the two functions the index of 0 corresponds to a shift of halft of a's length to the left.
    # No shift of a is to be expected at half of a's length...
    return np.correlate(arr, a, mode = "valid")


def moving_average(data, windowsize, stepsize = 1):
    '''
    description:
        calculate the moving average of a dataset with a given windowsize.

    inputs:
        data             -  The dataset, that will be averaged
        windowsize       -  The number of datapoints that will be used to calculate the average
        stepsize         -  The number of datapoints between each window

    returns:
        avg_data         -  The moving average
    '''
    if stepsize < 1 or stepsize > windowsize:
        print("When calculating a moving average: The stepsize should be between 1 and the windowsize")
    else:
        r = range(0, len(data)-windowsize, stepsize)
        number_of_windows = len(r)
        avg_data = np.zeros(number_of_windows, dtype =np.complex64) # allocate memory
        for idx, n in enumerate(r):
            avg_data[idx] = np.mean(data[n:n + windowsize])
        return np.array(avg_data)

def cyclic_moving_average(data, windowsize):
    '''
    description:
        calculate the moving average of a dataset with a given windowsize. At the
        edges of the dataset, it will be continued cyclically.

    inputs:
        data             -  The dataset, that will be averaged
        windowsize       -  The number of datapoints that will be used to calculate the average

    returns:
        avg_data         -  The cyclically calculated moving average
    '''
    avg_data = np.zeros(np.shape(data), dtype =np.complex64) #allocate memory
    for n in range(len(data)):
        # shift the datapoints in a way, that the window of interest is in the beginning of the set
        avg_data[n] = np.mean(np.roll(data,-n+int(windowsize/2))[:windowsize])
    return avg_data


def devide_in_batches(data, batchsize):
    '''
    description:
        slice the raw measurement data into batches

    inputs:
        data             -  The array of measurement points
        batchsize        -  The number of datapoints per batch

    returns:
        batches          -  A 2D-Array containing the batches of the measurement.
                            The axis represent: 0 - batchnumber; 1 - time index in batch
    '''
    if len(data) % batchsize:
        print("There has been an over or underflow. Please repeat the measurement!") # maybe raise an exception
    number_of_batches = int( (len(data)) / batchsize )
    return np.reshape(data, (number_of_batches, batchsize)) # rearrange into 2D-Array


def correlate_batches(batches):
    '''
    description:
        correlate the raw data batches with the zc-zc_sequence

    inputs:
        batches          -  A 2D-Array containing the batches of the measurement.
                            The axis represent: 0 - batchnumber; 1 - time index in batch

    returns:
        corr             -  A 3D-Array. The three axis represent:
                            0 - batchnumber; 1 - index of impulse response in batch; 2 - time delay of impulse response
    '''
    corr = batches
    for idx,c in enumerate(corr.copy()):
        c = np.correlate(c, zc_seq, mode = "same")/batchsize # correlate every batch with zc-sequence
        corr[idx] = c
    corr = np.reshape(corr, (number_of_batches, int(round(batchsize/zc_len)), zc_len)) # rearrange into 3D-array
    return corr


def calculate_timevariant_transferfunction(h, usable_bandwidth = 20e6):
    '''
    description:
        caclulate the timevariant transferfunction with a cutoff frequency of usable_bandwidth/2

    inputs:
        h                -  The impulse response of the measurement. A 2D-Array the axis reptrsent:
                            0 - batchnumber; 1 - time delay of impulse response
        usable_bandwidth -  Determines the cutoff frequecy to lose lowpass effects

    returns:
        T                -  The timevariant transferfunction. A 2D-Array: the two axis represent:
                            0 - Batchnumber; 1 - Frequency bins of the batch
    '''
    T = np.zeros(np.shape(h),dtype = np.complex64) # timevariant transferfunction
    for idx_h, h_ in enumerate(h):
        T[idx_h] = np.fft.fftshift(np.fft.fft(h_, norm = "ortho")) # fft
    #Cut Lowpass filter effects away to 20 MHz
    if usable_bandwidth < fs:
        # calculate cut-off bins
        unusable_bins = int(round(np.shape(h)[1]/2 * (1-usable_bandwidth/fs)))
        T = T[:, unusable_bins:-unusable_bins]
    return T

def cfo_correction(corr, upsample_factor):
    '''
    description:
        Perfom a cfo correction to a measurement

    inputs:
        corr             -  A 3D-Array. The three axis represent:
                            0 - batchnumber; 1 - index of impulse response in batch; 2 - time delay of impulse response
        upsample_factor  -  The factor by which the batchsize will be increased

    returns:
        corrected_corr   -  A 3D-Array. The three axis represent:
                            0 - batchnumber; 1 - index of impulse response in batch; 2 - time delay of impulse response
                            The impulse responses are cfo corrected.
    '''
    batchsize = int(np.shape(corr)[1] * np.shape(corr)[2]) # batchsize = number of impulse responses * length of impulse responses
    corr_ = np.reshape(corr, (np.shape(corr)[0], batchsize)) # rearrange data in 2D-Array
    corrected_corr = np.zeros(np.shape(corr_), dtype =np.complex64)
    for idx, x in enumerate(corr_.copy()): # for every batch:
        upsampled_function = resample(x, len(x) * upsample_factor) # Perform sinc interpolation
        h_max = []
        for offset in range(upsample_factor): # find the offset value with the highest maximum value
            resampled_axis = np.rint(np.arange(offset, len(upsampled_function)-1, upsample_factor)).astype(int)
            h_max.append(np.max(abs(upsampled_function[resampled_axis])))
        offset = np.argmax(h_max)
        resampled_axis = np.rint(np.arange(offset, len(upsampled_function)-1, upsample_factor)).astype(int) #create axis for downsampling
        new_h = upsampled_function[resampled_axis] # downsample the data
        if len(new_h) >len(x): # crop the function at the end, if its too long
            new_h = new_h[:len(x)]
        if len(new_h) < len(x): # add 0s to the beginning of the function, if it's too short
            new_h = np.pad(new_h, (0,len(x) - len(new_h)), 'constant')
        corrected_corr[idx] = new_h
    corrected_corr = np.reshape(corrected_corr, np.shape(corr)) # rearrange back into a 3D-Array
    for idx_batch,batch in enumerate(corrected_corr.copy()): # Allign all maximums of the impulse responses
        for idx_response, response in enumerate(batch):
            shift = -np.argmax(abs(response)) + 10 # All maximums will be alligned at this index
            response = np.roll(response, shift)
            corrected_corr[idx_batch, idx_response] = response
    return corrected_corr

def calculate_impulse_response(corr):
    '''
    description:
        average the batches to calculate the impulse response

    inputs:
        corr:            -  A 3D-Array. The three axis represent:
                            0 - batchnumber; 1 - index of impulse response in batch; 2 - time delay of impulse response

    returns:
        h:               -  The impulse response of the measurement. A 2D-Array the axis reptrsent:
                            0 - batchnumber; 1 - time delay of impulse response
    '''
    corr = corr[:,1:-1,:] # Cut away edge effects of the batches
    return np.mean(corr, axis = 1) # average over the multiple impulse responses per batch

def calculate_B_coh(T, threshold, mode = "auto"):
    '''
    description:
        Calculate the coherence bandwidth as a function of time.

    inputs:
        T                -  The timevariant transferfunction. A 2D-Array: the two axis represent:
                            0 - Batchnumber; 1 - Frequency bins of the batch
        threshold        -  The threshold value as a factor to the maximum value of the autocorrelation function
                            up to which the function is still considered coherent
        mode             -  calculate B_coh either with autocorrelation function("auto") or
                            the cyclic corelation function ("cyclic")

    returns:
        B_coh            - The coherence bandwith as a function of time
    '''

    if not 0<=threshold<=1:
        print("When calculating B_coh: threshold value has to be between 0 and 1")
    B_coh = np.zeros(np.shape(T)[0])
    for idx_t, t in enumerate(T):
        freq_corr_function = []
        if mode == "cyclic":
            func = t - cyclic_moving_average(t, int(windowsize_in_sec/capture_interval))
            freq_corr_function = np.fft.fftshift( cyclic_auto_correlate(func))
        if mode == "auto":
            freq_corr_function = np.fft.fftshift(np.correlate(t,t,mode = "full"))
        freq_corr_max = abs( np.max(freq_corr_function) )

        x =  np.argmax( abs(freq_corr_function) < threshold * freq_corr_max )
        if np.min(abs(freq_corr_function)) > threshold * freq_corr_max :
            B_coh[idx_t] = np.min([fs, 20e6])
        else:
            B_coh[idx_t] = x * 2*(np.min([fs, 20e6]))/len(freq_corr_function)
    return B_coh


def calculate_T_coh(T, threshold, mode = "auto"):
    '''
    description:
        Calculate the coherence time as a function of time

    inputs:
        T                -  The timevariant transferfunction. A 2D-Array: the two axis represent:
                            0 - Batchnumber; 1 - Frequency bins of the batch
        threshold        -  The threshold value as a factor to the maximum value of the autocorrelation function
                            up to which the function is still considered coherent
        mode             -  calculate T_coh either with autocorrelation function("auto"),
                            the cyclic corelation function ("cyclic") or the sliding window method ("slide")
                            (picking a time window and slide and correlate it against the original function)

    returns:
        T_coh            - The coherence time as a function of time
    '''

    T = np.swapaxes(T,0,1)
    r = range(0, int(number_of_batches-batches_per_value),int(windowsize_in_sec/(2*capture_interval)))
    T_coh = np.zeros(len(r))
    for idx_t, t in enumerate(r):
        T_coh_f = np.zeros(np.shape(T)[0])
        for idx_f, f in enumerate(T):
            time_corr_max = 0
            time_corr_function = []

            if mode == "auto":
                time_corr_function = np.fft.fftshift( np.correlate( f[t:t+batches_per_value], f[t:t+batches_per_value], mode = "full" ) )
                time_corr_max = abs(np.max(time_corr_function))

            if mode == "cyclic":
                func = f[t:t+batches_per_value] - cyclic_moving_average(f[t:t+batches_per_value], int(windowsize_in_sec/capture_interval))
                time_corr_function = np.fft.fftshift( cyclic_auto_correlate(func) )
                time_corr_max = abs( np.max(time_corr_function) )

            if mode == "slide":
                window = f[t:t+batches_per_value] - np.mean(f[t:t+batches_per_value])
                function_window = np.roll(f, -t+batches_per_value)[0:3*batches_per_value]
                function_window = function_window - cyclic_moving_average(function_window, batches_per_value)
                time_corr_function = np.fft.fftshift(np.correlate(function_window,window, mode = "full"))
                time_corr_max = time_corr_function[int(len(time_corr_function)/2)]

            coherece_time = ( np.argmax( abs(time_corr_function) < threshold * time_corr_max ))
            if np.min(abs(time_corr_function)) > threshold * time_corr_max :
                T_coh_f[idx_f] = windowsize_in_sec
            else:
                T_coh_f[idx_f] = min(coherece_time * 2*windowsize_in_sec/len(time_corr_function), windowsize_in_sec)
        T_coh[idx_t] = np.mean(T_coh_f)

    return T_coh





def calculate_B_coh_power_delay_spread(h, threshold):
    '''
    description:
        Calculate the coherence bandwidth with by estimating it with thepower delay spread

    inputs:
        h                -  The impulse response of the measurement. A 2D-Array the axis reptrsent:
                            0 - batchnumber; 1 - time delay of impulse response
        threshold        -  The threshold value as a factor to the maximum value of the autocorrelation function
                            up to which the function is still considered coherent

    returns:
        B_coh            - The coherence bandwith as a function of time
    '''
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
    windowsize_in_sec = 1
    batches_per_value = round(windowsize_in_sec/capture_interval)
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

        B_coh_50 = calculate_B_coh(T, 0.5, mode = "cyclic")
        B_coh_90 = calculate_B_coh(T, 0.9, mode = "cyclic")

        signal_time = capture_interval * number_of_batches
        time_of_movavg_filter = batches_per_value * capture_interval
        time = np.linspace(time_of_movavg_filter/2, signal_time-time_of_movavg_filter/2, len(B_coh_50))

        plot_B_coh(B_coh_50, B_coh_90, time, np.min([fs,20e6]))

        B_coh_50_mean, B_coh_50_conf_interval_999 = confidence_interval(B_coh_50, 0.999)
        B_coh_90_mean, B_coh_90_conf_interval_999 = confidence_interval(B_coh_90, 0.999)
        print(f'99.9% of values of B_coh_50 lie in between {B_coh_50_conf_interval_999}[Hz]')
        print(f'99.9% of values of B_coh_90 lie in between {B_coh_90_conf_interval_999}[Hz]')

        #T_coh


        T_coh_50 = calculate_T_coh(T, 0.5, mode = "slide")
        T_coh_90 = calculate_T_coh(T, 0.9, mode = "slide")

        time = np.linspace(windowsize_in_sec/2, signal_time - windowsize_in_sec/2, len(T_coh_50))

        plot_T_coh(T_coh_50, T_coh_90, time, windowsize_in_sec)

        T_coh_50_mean, T_coh_50_conf_interval_999 = confidence_interval(T_coh_50, 0.999)
        T_coh_90_mean, T_coh_90_conf_interval_999 = confidence_interval(T_coh_90, 0.999)
        print(f'99.9% of values of T_coh_50 lie in between {T_coh_50_conf_interval_999}[s]')
        print(f'99.9% of values of T_coh_90 lie in between {T_coh_90_conf_interval_999}[s]')
        print("")
        plt.show()
