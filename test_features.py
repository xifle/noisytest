import matplotlib.pyplot as plt
import tikzplotlib
import tensorflow as tf
import numpy as np
from ac_import import *
from ac_preprocess import *


def center_data(x):
    return mean_subtract.center_data(x)


def plot_time_data(x):
    plt.plot(np.linspace(0, ac_config.samplesPerFrame/ac_config.inputDataSampleRate, ac_config.samplesPerFrame), x)
    plt.xlabel("Time [s]")
    plt.ylabel("$\\hat p$")
    plt.grid(True)


def plot_spectrogram(x):
    plt.imshow(tf.transpose(x), aspect='auto')
    plt.xlabel('Time [$\\frac{s_{fft}^*}{f_s}$]')
    plt.ylabel('Frequency [$\\frac{f_s}{n_{fft}}$]')


mean_subtract = MeanSubtraction()
(input, output) = read_training_data(True, False)
specs = prepare_compressed(input)
mean_subtract.learn(specs, True)

(data_impact, label_impact, u, v) = read_experiment('training/jointdecalibration', True, True)

idx = 2

plot_time_data(data_impact[idx])
tikzplotlib.save('export/time_data.tex')
plt.show()

plot_spectrogram(mag2log(spectrogram(data_impact[idx])))
tikzplotlib.save('export/spectrogram.tex')
plt.show()

plot_spectrogram(mag2log(compress_spectrogram(spectrogram(data_impact[idx]), 3)))
tikzplotlib.save('export/spectrogram_compressed.tex')
plt.show()

plot_spectrogram(mean_subtract.center_data(prepare_compressed(data_impact[idx])))
tikzplotlib.save('export/final.tex')
plt.show()
