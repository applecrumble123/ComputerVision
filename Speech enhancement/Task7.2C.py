import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display as ipd
from pydub import AudioSegment
from pydub.utils import mediainfo
import os
from scipy import signal
import array
import pydub
from pydub import AudioSegment

ROOTDIR = '/Users/johnathontoh/Desktop/Task 7.2/Resources_7.2'

def get_sample_and_frame_rate(filepath):
    speech = AudioSegment.from_wav(filepath)

    # samples x(t)
    samples = speech.get_array_of_samples()

    # sampling rate f - see slide 24 in week 7 lecture slides
    sampling_rate = speech.frame_rate

    return speech, samples, sampling_rate


def visualise_audio(samples):
    plt.figure(figsize = (15, 5))
    plt.plot(samples)
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.show()


def spectrogram(samples, sampling_rate):
    #range of frequencies of interest for speech signal.
    #It can be any positive value, but should be a power of 2
    freq_range = 2048

    #window size: the number of samples per frame #each frame is of 30ms
    win_length = int(sampling_rate * 0.03)

    #number of samples between two consecutive frames #by default, hop_length = win_length / 4
    hop_length = int(win_length / 2)

    #windowing technique
    window = 'hann'

    noisy_S = librosa.stft(np.float32(samples), n_fft = freq_range, window = window, hop_length = hop_length, win_length = win_length)

    plt.figure(figsize = (15, 5))


    #convert the amplitude to decibels, just for illustration purpose
    noisy_Sdb = librosa.amplitude_to_db(abs(noisy_S))
    librosa.display.specshow(   #spectrogram
                                noisy_Sdb,

                                #sampling rate
                                sr = sampling_rate,

                                #label for horizontal axis
                                x_axis = 'time',

                                #presentation scale
                                y_axis = 'linear',

                                #hop_length
                                hop_length = hop_length)

    plt.show()



def filter(order, cutoff_freq, filter_type, sampling_rate):
    #order
    order = order

    #sampling frequency
    sampling_freq = sampling_rate

    # cut-off frequency. This can be an array if band-pass filter is used
    #cthis must be within 0 and cutoff_freq/2
    cutoff_freq = cutoff_freq

    #filter type, e.g., 'lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’
    filter_type = filter_type

    #filter
    filter = signal.butter(N = order,
                    fs = sampling_freq,
                    Wn = cutoff_freq,
                    btype = filter_type,
                    analog = False,
                    output = 'sos')
    return filter



def get_filtered_audio(filter, samples, sampling_rate, speech):
    filtered_s = signal.sosfilt(filter, samples)

    filtered_s_audio = pydub.AudioSegment( #raw data
                                            data = array.array(speech.array_type, np.float16(filtered_s)),

                                            #2 bytes = 16 bit samples
                                            sample_width = 2,

                                            #frame rate
                                            frame_rate = sampling_rate,

                                            #channels = 1 for mono and 2 for stereo
                                            channels = 1)
    return filtered_s_audio


# original noisy sound
noisy_speech, noisy_s, noisy_f = get_sample_and_frame_rate(os.path.join(ROOTDIR,'NoisySignal/Station/sp01_station_sn5.wav'))
visualise_audio(noisy_s)
spectrogram(noisy_s, noisy_f)

#---------------------------- lowpass filter --------------------------------------

# filter
filter_lowpass = filter(order=10, cutoff_freq=1000, filter_type='lowpass', sampling_rate = noisy_f)
filtered_s_audio_lowpass = get_filtered_audio(filter_lowpass, noisy_s, noisy_f, noisy_speech)

# export the sound
filtered_s_audio_lowpass.export(os.path.join(ROOTDIR,'sp01_station_sn5_lowpass.wav'), format = 'wav')

# get the speech, sample and sampling rate of the filtered sound
noisy_lowpass_speech, noisy_lowpass_s, noisy_lowpass_f = get_sample_and_frame_rate(os.path.join(ROOTDIR,'sp01_station_sn5_lowpass.wav'))
# get the spectrogram of the filtered sound
spectrogram(noisy_lowpass_s, noisy_lowpass_f)

#---------------------------- highpass filter --------------------------------------

# filter
filter_highpass = filter(order=10, cutoff_freq=200, filter_type='highpass', sampling_rate = noisy_f)
filtered_s_audio_highpass = get_filtered_audio(filter_highpass, noisy_s, noisy_f, noisy_speech)

# export the sound
filtered_s_audio_highpass.export(os.path.join(ROOTDIR,'sp01_station_sn5_highpass.wav'), format = 'wav')

# get the speech, sample and sampling rate of the filtered sound
noisy_highpass_speech, noisy_highpass_s, noisy_highpass_f = get_sample_and_frame_rate(os.path.join(ROOTDIR,'sp01_station_sn5_highpass.wav'))
# get the spectrogram of the filtered sound
spectrogram(noisy_highpass_s, noisy_highpass_f)


#---------------------------- bandpass filter --------------------------------------
# filter
filter_bandpass = filter(order=10, cutoff_freq=[200, 1000], filter_type='bandpass', sampling_rate = noisy_f)
filtered_s_audio_bandpass = get_filtered_audio(filter_bandpass, noisy_s, noisy_f, noisy_speech)

# export the sound
filtered_s_audio_bandpass.export(os.path.join(ROOTDIR,'sp01_station_sn5_bandpass.wav'), format = 'wav')

# get the speech, sample and sampling rate of the filtered sound
noisy_bandpass_speech, noisy_bandpass_s, noisy_bandpass_f = get_sample_and_frame_rate(os.path.join(ROOTDIR,'sp01_station_sn5_bandpass.wav'))
# get the spectrogram of the filtered sound
spectrogram(noisy_bandpass_s, noisy_bandpass_f)


def get_magnitude(filepath):
    # Read audio data from file
    speech = AudioSegment.from_wav(filepath)
    samples = speech.get_array_of_samples() # samples x(t)
    sampling_rate = speech.frame_rate # sampling rate f - see slide 24 in week 7 lecture slides

    #window size: the number of samples per frame #each frame is of 30ms
    win_length = int(sampling_rate * 0.03)
    # number of samples between two consecutive frames #by default, hop_length = win_length / 4
    hop_length = int(win_length / 2)
    Y_or_D = librosa.stft(np.float32(samples), n_fft=2048, window = 'hann', hop_length = hop_length, win_length = win_length)
    magnitude = abs(Y_or_D)

    return magnitude, Y_or_D, samples, sampling_rate, speech

def fourier_transform(mag_Y, means_mag_D, condition):
    H = np.zeros((mag_Y.shape[0], mag_Y.shape[1]), np.float32)
    for k in range(H.shape[0]):
        for t in range(H.shape[1]):
            weiner_filer = condition
            if weiner_filer is False:
                H[k][t] = np.sqrt(max(0, 1 - (means_mag_D[k] * means_mag_D[k]) / (mag_Y[k][t] * mag_Y[k][t])))
                print("Spectral Subtraction")
            else:
                H[k][t] = max(0, 1 - (means_mag_D[k] * means_mag_D[k]) / (mag_Y[k][t] * mag_Y[k][t]))
                print("Weiner Filter")
    return H

def S_hat_and_s_hat(mag_Y, H, Y, sampling_rate, samples):
    S_hat = np.zeros((mag_Y.shape[0], mag_Y.shape[1]), np.float32)
    for k in range(H.shape[0]):
        for t in range(H.shape[1]):
            S_hat[k][t] = H[k][t] * Y[k][t]

    win_length = int(sampling_rate * 0.03)
    hop_length = int(win_length / 2)
    s_hat = librosa.istft(S_hat, win_length = win_length, hop_length = hop_length, length = len(samples))

    return s_hat

def export_audio(speech_y, s_hat, sampling_rate, filename):
    audio = AudioSegment( #raw data
                                data = array.array(speech_y.array_type, np.float16(s_hat)),

                                #2 bytes = 16 bit samples
                                sample_width = 2,

                                #frame rate
                                frame_rate = sampling_rate,

                                #channels = 1 for mono and 2 for stereo
                                channels = 1)

    audio.export(os.path.join(ROOTDIR, filename), format = 'wav')


mag_Y, Y, y, y_f, speech_y = get_magnitude(os.path.join(ROOTDIR,'NoisySignal/Station/sp01_station_sn5.wav'))

mag_D, D, d, d_f, speech_d = get_magnitude(os.path.join(ROOTDIR,'Noise/Station/Station_1.wav'))

means_mag_D = np.mean(mag_D, axis = 1)

# True is for weiner_filter, False it means the filter is not used
H = fourier_transform(mag_Y, means_mag_D, False)

s_hat = S_hat_and_s_hat(mag_Y, H, Y, y_f, y)


export_audio(speech_y, s_hat, y_f, 'sp01_station_sn5_spectralsubtraction.wav')

# get the speech, sample and sampling rate of the filtered sound
speech_spectralsubtraction_noisy, s_spectralsubtraction_noisy, f_spectralsubtraction_noisy = get_sample_and_frame_rate(os.path.join(ROOTDIR,'sp01_station_sn5_spectralsubtraction.wav'))
# get the spectrogram of the filtered sound
spectrogram(s_spectralsubtraction_noisy, f_spectralsubtraction_noisy)


# get the speech, sample and sampling rate of the filtered sound
speech_clean, s_clean, f_clean = get_sample_and_frame_rate(os.path.join(ROOTDIR,'CleanSignal/sp01.wav'))
# get the spectrogram of the filtered sound
spectrogram(s_clean, f_clean)


CLEAN_SIGNAL_DIR = '/Users/johnathontoh/Desktop/Task 7.2/Resources_7.2/CleanSignal'

BABBLE_DIR = '/Users/johnathontoh/Desktop/Task 7.2/Resources_7.2/NoisySignal/Babble'

NOISE_BABBLE_DIR = '/Users/johnathontoh/Desktop/Task 7.2/Resources_7.2/Noise/Babble'

SAVED_SPECTRAL_SUBTRACTION_DIR = '/Users/johnathontoh/Desktop/Task 7.2/Resources_7.2/spectral subtraction/Babble'

SAVED_WEINER_FILTER_DIR = '/Users/johnathontoh/Desktop/Task 7.2/Resources_7.2/weiner filter/Babble'

"""
# get spectrogram for each clean noise
for i in os.listdir(CLEAN_SIGNAL_DIR):
    speech_clean, s_clean, f_clean = get_sample_and_frame_rate(os.path.join(CLEAN_SIGNAL_DIR, i))
    # get the spectrogram of the filtered sound
    spectrogram(s_clean, f_clean)
"""

mag_D_babble, D_babble, d_babble, d_f_babble, speech_d_babble = get_magnitude(os.path.join(NOISE_BABBLE_DIR, 'Babble_1.wav'))

"""
count_ss = 1
# to get the spectral subtraction
for i in os.listdir(BABBLE_DIR):
    speech_noisy_babble_ss, s_noisy_babble_ss, f_noisy_babble_ss = get_sample_and_frame_rate(os.path.join(BABBLE_DIR, i))
    
    # get the spectrogram of the filtered sound
    #spectrogram(s_noisy_babble_ss, f_noisy_babble_ss)

    mag_Y_babble_ss, Y_babble_ss, y_babble_ss, y_f_babble_ss, speech_y_babble_ss = get_magnitude(os.path.join(BABBLE_DIR, i))

    means_mag_D_babble_ss = np.mean(mag_D_babble, axis=1)

    # True is for weiner_filter, False it means the filter is not used
    H_babble_spectral_subtraction = fourier_transform(mag_Y_babble_ss, means_mag_D_babble_ss, False)

    s_hat_babble_ss = S_hat_and_s_hat(mag_Y_babble_ss, H_babble_spectral_subtraction, Y_babble_ss, y_f_babble_ss, y_babble_ss)

    export_audio(speech_y_babble_ss, s_hat_babble_ss, y_f_babble_ss, os.path.join(SAVED_SPECTRAL_SUBTRACTION_DIR, 'sp0' + str(count_ss) + '_babble_sn5_spectralsubtraction.wav' ))

    count_ss = count_ss + 1
"""

"""
# to get the spectrogram for spectral subtraction
for i in os.listdir(SAVED_SPECTRAL_SUBTRACTION_DIR):
    speech_spectral_subtraction, s_spectral_subtraction, f_spectral_subtraction = get_sample_and_frame_rate(os.path.join(SAVED_SPECTRAL_SUBTRACTION_DIR, i))
    # get the spectrogram of the filtered sound
    spectrogram(s_spectral_subtraction, f_spectral_subtraction)
"""

"""

count_wf = 1
for i in os.listdir(BABBLE_DIR):
    speech_noisy_babble_wf, s_noisy_babble_wf, f_noisy_babble_wf = get_sample_and_frame_rate(os.path.join(BABBLE_DIR, i))

    # get the spectrogram of the filtered sound
    # spectrogram(s_noisy_babble, f_noisy_babble)

    mag_Y_babble_wf, Y_babble_wf, y_babble_wf, y_f_babble_wf, speech_y_babble_wf = get_magnitude(os.path.join(BABBLE_DIR, i))

    means_mag_D_babble_wf = np.mean(mag_D_babble, axis=1)

    # True is for weiner_filter, False it means the filter is not used
    H_babble_wf = fourier_transform(mag_Y_babble_wf, means_mag_D_babble_wf, True)

    s_hat_babble_wf = S_hat_and_s_hat(mag_Y_babble_wf, H_babble_wf, Y_babble_wf, y_f_babble_wf, y_babble_wf)

    export_audio(speech_y_babble_wf, s_hat_babble_wf, y_f_babble_wf, os.path.join(SAVED_WEINER_FILTER_DIR, 'sp0' + str(count_wf) + '_babble_sn5_weinerfilter.wav'))

    count_wf = count_wf + 1
"""


"""
# to get the spectrogram for spectral subtraction
for i in os.listdir(SAVED_WEINER_FILTER_DIR):
    speech_wf, s_wf, f_wf = get_sample_and_frame_rate(os.path.join(SAVED_WEINER_FILTER_DIR, i))
    # get the spectrogram of the filtered sound
    spectrogram(s_wf, f_wf)
"""