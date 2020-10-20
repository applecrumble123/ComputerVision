import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display as ipd
from pydub import AudioSegment
from pydub.utils import mediainfo
import os

ROOTDIR = "/Users/johnathontoh/Desktop/Task 7.1/Resources_7.1"

def get_sample_and_frame_rate(filepath):
    # Read audio data from file
    speech = AudioSegment.from_wav(filepath)
    # samples x(t)
    x = speech.get_array_of_samples()
    # sampling rate f - see slide 24 in week 7 lecture slides
    x_sr = speech.frame_rate
    print('Sampling rate: ', x_sr)
    print('Number of samples: ', len(x))

    return x, x_sr

def visualise_audio_clips(x, x_sr, filepath):
    duration = librosa.get_duration(filename = filepath)
    n_samples = duration * x_sr
    print('duration: ', duration)
    print('n_samples: ', n_samples)


    x_range = np.linspace(0, duration, len(x))
    plt.figure(figsize = (15, 5))
    plt.plot(x_range, x)
    plt.xlabel('Time in seconds')
    plt.ylabel('Amplitude')
    plt.show()


original_x, original_x_sr = get_sample_and_frame_rate(os.path.join(ROOTDIR,'arctic_a0005.wav' ))

visualise_audio_clips(original_x, original_x_sr, os.path.join(ROOTDIR,'arctic_a0005.wav' ))

mid_point = int(len(original_x) / 2)
x1 = original_x[0:mid_point]
x2 = original_x[mid_point:len(original_x)]

x1_audio = AudioSegment( #raw data
                        data = x1,
                        #2 bytes = 16 bit samples
                        sample_width = 2,
                        #frame rate
                        frame_rate = original_x_sr,
                        #channels = 1 for mono and 2 for stereo
                        channels = 1)

x2_audio = AudioSegment( #raw data
                        data = x2,
                        #2 bytes = 16 bit samples
                        sample_width = 2,
                        #frame rate
                        frame_rate = original_x_sr,
                        #channels = 1 for mono and 2 for stereo
                        channels = 1)

x1_audio.export(os.path.join(ROOTDIR,'arctic_a0005_1.wav'), format = 'wav')
x2_audio.export(os.path.join(ROOTDIR,'arctic_a0005_2.wav'), format = 'wav')



x1, x1_sr = get_sample_and_frame_rate(os.path.join(ROOTDIR,'arctic_a0005_1.wav'))

visualise_audio_clips(x1, x1_sr, os.path.join(ROOTDIR,'arctic_a0005_1.wav'))


x2, x2_sr = get_sample_and_frame_rate(os.path.join(ROOTDIR,'arctic_a0005_2.wav'))

visualise_audio_clips(x2, x2_sr, os.path.join(ROOTDIR,'arctic_a0005_2.wav'))


def spectogram(x, x_sr):
    #range of frequencies of interest for speech signal.
    #It can be any positive value, but should be a power of 2
    freq_range = 1024

    #window size: the number of samples per frame
    #each frame is of 30ms
    win_length = int(x_sr * 0.03)

    #number of samples between tww consecutive frames
    hop_length = int(win_length / 2)

    #windowing technique
    window = 'hann'

    #bydefault, hop_length = win_length / 4
    X = librosa.stft(np.float32(x),
                     n_fft = freq_range,
                     window = window,
                     hop_length = hop_length,
                     win_length = win_length)

    print(X.shape)

    plt.figure(figsize = (15, 5))
    #convert the amplitude to decibels, just for illustration purpose
    Xdb = librosa.amplitude_to_db(abs(X))
    librosa.display.specshow(   #spectrogram
                                Xdb,
                                #sampling rate
                                sr = x_sr,
                                #label for horizontal axis
                                x_axis = 'time',
                                #presentation scale
                                y_axis = 'linear',
                                #hop_lenght
                                hop_length = hop_length)


    plt.show()

spectogram(original_x, original_x_sr)

spectogram(x1, x1_sr)

spectogram(x2, x2_sr)

#number of samples
N = 600

#sample spacing
T = 1.0 / 600.0
t = np.linspace(0.0, N*T, N)
s1 = np.sin(50.0 * 2.0 * np.pi * t)
s2 = 0.5 * np.sin(80.0 * 2.0 * np.pi * t)
s = s1 + s2

plt.figure(figsize = (15, 5))
plt.plot(s1, label = 's1', color = 'r')
plt.plot(s2, label = 's2', color = 'g')
plt.plot(s, label = 's', color = 'b')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.legend(loc = "upper left")
plt.show()

def sinusoidal_signal(window_technique):
    S = librosa.stft(s, n_fft = 1024, window = window_technique, hop_length = N, win_length = N)
    S_0 = S[:, 0]
    mag_S_0 = np.abs(S_0)
    plt.plot(mag_S_0, color = 'b')
    plt.show()

    #we define a window length m with less number of samples
    m = 400

    S = librosa.stft(s, n_fft = 1024, window = window_technique, hop_length = int(m / 2), win_length = m)

    #we take S_1, which is an intermediate frame.
    S_1 = S[:, 1]

    mag_S_1 = np.abs(S_1)

    plt.plot(mag_S_1, color = 'b')

    plt.show()

#sinusoidal_signal('hann')
#sinusoidal_signal('boxcar')
sinusoidal_signal('hamming')
