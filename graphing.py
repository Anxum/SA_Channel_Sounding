import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_recieved_power(recieved_power_dbfs, time):
    plt.figure(num="P(t)")
    plt.plot(time, recieved_power_dbfs)
    plt.xlabel("Time [s]")
    plt.ylabel("Power of detected signal [dBFS]")
    plt.ylim([-60, 0])
    plt.title("Received signal power over time")

def plot_impulse_response(h, time, delay):
    tau,t = np.meshgrid(delay, time)
    fig = plt.figure(num = "h(t,delta_t)")
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Delay [s]")
    ax.set_zlabel("h(t,delta_t)")
    ax.plot_surface(t, tau, abs(h), cmap="plasma")

def plot_timevariant_transferfunction(T, time, frequency):
    f, t = np.meshgrid(frequency, time)
    fig = plt.figure(num = "T(t,f) [dBFS]")
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_zlabel("T(t,f) [dBFS]")
    ax.plot_surface(t, f, 10*np.log10(abs(T)), cmap="plasma")

def plot_B_coh(b_50, b_90, time, max_y):
    plt.figure(num="B_coh(t)")
    plt.plot(time, b_50, "-b", label = 'B_coh_50%(t)' )
    plt.plot(time, b_90, "-r", label = 'B_coh_90%(t)' )
    plt.xlabel("Time [s]")
    plt.ylabel("B_coh [Hz]")
    plt.title("Coherence bandwidth over time")
    plt.ylim([0, 1.02*max_y])
    plt.legend(loc="best")

def plot_T_coh(t_50, t_90, time, max_y):
    plt.figure(num="T_coh(t)")
    plt.plot(time, t_50, "-b", label = 'T_coh_50%(t)' )
    plt.plot(time, t_90, "-r", label = 'T_coh_90%(t)' )
    plt.xlabel("Time [s]")
    plt.ylabel("T_coh [s]")
    plt.ylim([0, 1.02 * max_y])
    plt.title("Coherence times over time")
    plt.legend(loc="best")
