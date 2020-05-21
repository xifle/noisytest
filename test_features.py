import matplotlib.pyplot as plt
import tikzplotlib
import tensorflow as tf
import numpy as np
from ac_import import *
from ac_preprocess import *


def plot_time_data(x):
    plt.plot(np.linspace(0, ac_config.samplesPerFrame/ac_config.inputDataSampleRate, ac_config.samplesPerFrame), x)
    plt.xlabel("Time [s]")
    plt.ylabel("$\\hat p$")
    plt.grid(True)


def plot_spectrogram(x):
    print("shape: ", tf.shape(x))
    plt.imshow(tf.transpose(x), aspect='auto', interpolation='nearest')
    #plt.colorbar()
    plt.xlabel('Time [$\\frac{s_{\\mathrm{fft}}^*}{f_s}$]')
    plt.ylabel('Frequency [$\\frac{f_s}{n_{\\mathrm{fft}}}$]')


def plot_dctresult(x):
    print("shape: ", tf.shape(x))
    fig, ax = plt.subplots(1,1)
    plt.imshow(x, aspect='auto', interpolation='nearest')
    #plt.colorbar()
    ax.set_xticks([0.0, 1.0, 2.0, 3.0])
    ax.set_xticklabels(['0', '1', '2', '3'])
    plt.xlabel('DCT coefficients temporal continuation')
    plt.ylabel("Frequency [$\\frac{f_s}{n_{\\mathrm{fft}}}$]")


(input, output) = read_training_data(True, False)

(data_impact, label_impact, u, v) = read_experiment('training/jointdecalibration', True, True)
(data_normal, label_normal, u, v) = read_experiment('training/sequence', True, True)

idx = 4

plot_time_data(data_impact[idx])
plt.savefig('export/time_data.pgf')
tikzplotlib.save('export/time_data.tex')
plt.show()

plot_spectrogram(mag2log(spectrogram(data_impact[idx])))
plt.savefig('export/spectrogram.pgf')
tikzplotlib.save('export/spectrogram.tex')
plt.show()

plot_spectrogram(mag2log(compress_spectrogram(spectrogram(data_impact[idx]), 3)))
plt.savefig('export/spectrogram_compressed.pgf')
tikzplotlib.save('export/spectrogram_compressed.tex')
plt.show()

plot_dctresult(apply_dct(mag2log(compress_spectrogram(spectrogram(data_impact[idx]), 3))))
plt.savefig('export/final.pgf')
tikzplotlib.save('export/final.tex')
plt.show()
