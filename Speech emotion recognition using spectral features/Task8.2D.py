import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display as ipd
from pydub import AudioSegment
from pydub.utils import mediainfo
import os, random
import math
import pickle

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble  import AdaBoostClassifier


from sklearn.svm import SVC
from sklearn.ensemble  import AdaBoostClassifier

ROOTDIR = '/Users/johnathontoh/Desktop/Task8.2D/Resources_8.2'

def extract_mel_spectogram(filepath, power, num_frames):
    # Read audio data from file
    speech = AudioSegment.from_wav(filepath)
    # samples x(t)
    samples = speech.get_array_of_samples()
    # sampling rate f - see slide 24 in week 7 lecture slides
    sampling_rate = speech.frame_rate

    #print('Sampling rate: ', sampling_rate)
    #print('Number of samples: ', len(samples))

    mel = librosa.feature.melspectrogram(   # number of samples
                                            y=np.float32(samples),

                                            #  frequency range of interest in mel-scale is [300, 3400]
                                            # 0 - 3400 has 3401 elements
                                            n_mels = 3401,

                                            # range of the Hert-scale frequencies
                                            # power of 2 but must be more than 10000 for hertz scale because the range is more than 3200
                                            # see slide 13 for lecture 8
                                            n_fft = 16384,

                                            # hop_length
                                            # sampling rate (i.e., frame rate) of input audio signal
                                            # 0.015 is the duration of the hop in seconds (i.e., 15ms).
                                            hop_length = int(sampling_rate*0.015),

                                            # power of the input signal's spectra in the mel scale
                                            power = power
                                            )

    #print(mel.shape)

    mel_truncated = np.zeros((mel.shape[0], num_frames), np.float32)
    # either get the frames (which is less than 200) or 200 frames
    # setting a limit of 200 frames
    for i in range(min(num_frames, mel.shape[1])):
        mel_truncated[:, i] = mel[:, i]

    #print(mel_truncated.shape)
    return samples, sampling_rate, mel, mel_truncated


def extract_SC(sub_bands, num_frames, mel_truncated):
    feature_vector = np.zeros([len(sub_bands), num_frames], np.float32)
    # looping through the sub bands
    for b in range(len(sub_bands)):
        # get the range of frequency in each frame
        for f in range(num_frames):
            # initialise the sum with a small number so the division will not be 0
            sum = 0.0000001
            # k refers to the frequency in the sub bands in a frame
            for k in range(sub_bands[b][0], sub_bands[b][1] + 1):
                # for the range of row (k) in mel_truncated (features) for each frame (f) for the mel_spectrogram (weights of the frequency)
                feature_vector[b][f] = feature_vector[b][f] + (k * mel_truncated[k][f])
                # sum up all the mel_spectrogram
                sum = sum + mel_truncated[k][f]

            # for all features in the feature vector in all frames of a sub_band
            feature_vector[b][f] = feature_vector[b][f]/sum

    vector_output = np.reshape(feature_vector.T, feature_vector.shape[0] * feature_vector.shape[1])

    return vector_output






def extract_SBW(sub_bands, num_frames, mel_truncated):
    feature_vector = np.zeros([len(sub_bands), num_frames], np.float32)
    # looping through the sub bands
    for b in range(len(sub_bands)):
        # get the range of frequency in each frame
        for f in range(num_frames):
            # initialise the sum with a small number so the division will not be 0
            sum = 0.0000001

            # get SC for each frame so the frequency
            SC = 0
            # k refers to the frequency in the sub bands in a frame
            for k in range(sub_bands[b][0], sub_bands[b][1] + 1):
                # for the range of row (k) in mel_truncated (features) for each frame (f) for the mel_spectrogram (weights of the frequency)
                SC = SC + (k * mel_truncated[k][f])
                # sum up all the mel_spectrogram
                sum = sum + mel_truncated[k][f]
            # get Spectral Centroid for each frame
            SC = SC/sum
            for k in range(sub_bands[b][0], sub_bands[b][1] + 1):
                feature_vector[b][f] = (k - SC)**2 * (mel_truncated[k][f])

            # # for all features in the feature vector in all frames of a sub_band, divide by the sum of mel_truncated in a sub_band in a frame
            feature_vector[b][f] = feature_vector[b][f]/sum

    vector_output = np.reshape(feature_vector.T, feature_vector.shape[0] * feature_vector.shape[1])

    return vector_output




def extract_SBE(sub_bands, num_frames, mel_truncated):
    feature_vector = np.zeros([len(sub_bands), num_frames], np.float32)

    # looping through the sub bands
    for b in range(len(sub_bands)):
        # get the range of frequency in each frame
        for f in range(num_frames):
            # initialise the sum with a small number so the division will not be 0
            # k refers to the frequency in the sub bands in a frame
            for k in range(sub_bands[b][0], sub_bands[b][1] + 1):
                # for the range of row (k) in mel_truncated (features) for each frame (f) for the mel_spectrogram (weights of the frequency)
                # sum of all mel_truncated[k][f] in a sub_band
                feature_vector[b][f] = feature_vector[b][f] + mel_truncated[k][f]

    # After the first part, each row (feature) in the feature vector has a sum of mel_truncated[k][f] respective to each frame

    for f in range(num_frames):
        # sum all the features for all the sub_band in that frame
        sum = np.sum(feature_vector[:, f]) + 0.0000001
        # for each feature in that frame, divide by the sum of all the sub_bands
        feature_vector[:, f] = feature_vector[:, f]/sum

    #print(feature_vector.shape)
    vector_output = np.reshape(feature_vector.T, feature_vector.shape[0] * feature_vector.shape[1])

    return vector_output



def extract_SFM(sub_bands, num_frames, mel_truncated):
    feature_vector = np.zeros([len(sub_bands), num_frames], np.float32)
    # looping through the sub bands
    for b in range(len(sub_bands)):
        # get the range of frequency in each frame
        p = 1 / (sub_bands[b][1] - sub_bands[b][0] + 1)
        for f in range(num_frames):
            # initialise the sum with a small number so the division will not be 0
            sum = 0.0000001
            # k refers to the frequency in the sub bands in a frame
            for k in range(sub_bands[b][0], sub_bands[b][1]):
                # for the range of row (k) in mel_truncated (features) for each frame (f) for the mel_spectrogram (weights of the frequency)
                feature_vector[b][f] = feature_vector[b][f] * np.power(mel_truncated[k][f], p)
                # sum up all the mel_spectrogram
                sum = sum + mel_truncated[k][f]

            # for all features in the feature vector in all frames of a sub_band, divide by (p x sum of all the mel_truncated)
            feature_vector[b][f] = feature_vector[b][f]/(p*sum)

    vector_output = np.reshape(feature_vector.T, feature_vector.shape[0] * feature_vector.shape[1])

    return vector_output



def extract_SCF(sub_bands, num_frames, mel_truncated):
    feature_vector = np.zeros([len(sub_bands), num_frames], np.float32)
    # looping through the sub bands
    for b in range(len(sub_bands)):
        # get the range of frequency in each frame
        p = 1 / (sub_bands[b][1] - sub_bands[b][0] + 1)
        for f in range(num_frames):
            # initialise the sum with a small number so the division will not be 0
            sum = 0.0000001
            # k refers to the frequency in the sub bands in a frame
            for k in range(sub_bands[b][0], sub_bands[b][1]):
                # sum up all the mel_spectrogram
                sum = sum + mel_truncated[k][f]
            # for all features in the feature vector in all the frames in a sub_band,
            # choose the max value for the mel_truncated in each range of sub_band for each frame
            feature_vector[b][f] = np.max(mel_truncated[sub_bands[b][0] : sub_bands[b][1]+1][f])/(p*sum)

    vector_output = np.reshape(feature_vector.T, feature_vector.shape[0] * feature_vector.shape[1])

    return vector_output





def extract_RE(sub_bands, num_frames, mel_truncated, alpha):
    feature_vector = np.zeros([len(sub_bands), num_frames], np.float32)
    # looping through the sub bands
    for b in range(len(sub_bands)):
        # get the range of frequency in each frame
        for f in range(num_frames):
            # sum of each sub_band in a frame
            # add the small value so the denominator will not be 0
            denominator = np.sum(abs(mel_truncated[sub_bands[b][0]: sub_bands[b][1]+1][f])) + 0.0000001

            # initialise the sum with a small number so there will not be log(0) which is undefined
            sum = 0.0000001

            # k refers to the frequency in the sub bands in a frame
            for k in range(sub_bands[b][0], sub_bands[b][1]):
                # sum of, mel_truncated in a sub_band divided by sum of mel_truncated for each frame in a sub_band
                sum = sum + np.power(abs(mel_truncated[k][f])/denominator, alpha)

            # for all features in the feature vector in all frames of a sub_band
            feature_vector[b][f] = (1/1-alpha)*(math.log2(sum))

    vector_output = np.reshape(feature_vector.T, feature_vector.shape[0] * feature_vector.shape[1])

    return vector_output





def extract_SE(sub_bands, num_frames, mel_truncated):
    feature_vector = np.zeros([len(sub_bands), num_frames], np.float32)
    # looping through the sub bands
    for b in range(len(sub_bands)):
        # get the range of frequency in each frame
        for f in range(num_frames):
            # sum of each sub_band in a frame
            # add the small value so the denominator will not be 0
            denominator = np.sum(mel_truncated[sub_bands[b][0]: sub_bands[b][1]+1][f]) + 0.0000001

            # initialise the sum with a small number so there will not be log(0) which is undefined
            sum = 0.0000001

            # k refers to the frequency in the sub bands in a frame
            for k in range(sub_bands[b][0], sub_bands[b][1]):
                # add the small value in log2 so there will not be log(0)
                sum = sum + (abs(mel_truncated[k][f]/denominator) * math.log2(abs(mel_truncated[k][f]/denominator) + 0.0000001))

            # for all features in the feature vector in all frames of a sub_band
            feature_vector[b][f] = - sum

    vector_output = np.reshape(feature_vector.T, feature_vector.shape[0] * feature_vector.shape[1])

    return vector_output




emotions = ['Calm', 'Happy', 'Sad', 'Angry']
path = os.path.join(ROOTDIR, 'EmotionSpeech/')

def get_train_name_and_label(filename):
    file_names = []
    emotion_labels = []
    for i in range(0, len(emotions)):
        sub_path = path + filename + '/' + emotions[i] + '/'
        sub_file_names = [os.path.join(sub_path, f) for f in os.listdir(sub_path)]
        sub_emotion_labels = [i] * len(sub_file_names)
        file_names += sub_file_names
        emotion_labels += sub_emotion_labels
    return file_names, emotion_labels




def spectrogram(mel, sampling_rate):

    #window size: the number of samples per frame #each frame is of 30ms
    win_length = int(sampling_rate * 0.015)

    #number of samples between two consecutive frames #by default, hop_length = win_length / 4
    hop_length = int(win_length / 2)

    plt.figure(figsize = (15, 5))


    #convert the amplitude to decibels, just for illustration purpose
    mel_db = librosa.amplitude_to_db(abs(mel))
    librosa.display.specshow(   #spectrogram
                                mel_db,

                                #sampling rate
                                sr = sampling_rate,

                                #label for horizontal axis
                                x_axis = 'time',

                                #presentation scale
                                y_axis = 'linear',

                                #hop_length
                                hop_length = hop_length)

    plt.show()






sub_bands  = [[300, 627], [628, 1060], [1061, 1633], [1634, 2393], [2394, 3400]]

"""
# on a given audio
samples_SC_SBW_SBE_SFM_SCF, sampling_rate_SC_SBW_SBE_SFM_SCF, mel_SC_SBW_SBE_SFM_SCF, mel_truncated_SC_SBW_SBE_SFM_SCF = extract_mel_spectogram(os.path.join(ROOTDIR, 'arctic_a0005.wav'), 2, 200)

# on a given audio
samples_RE_SE, sampling_rate_RE_SE, mel_RE_SE, mel_truncated_RE_SE = extract_mel_spectogram(os.path.join(ROOTDIR, 'arctic_a0005.wav'), 1, 200)

vector_output_SC = extract_SC(sub_bands, 200, mel_truncated_SC_SBW_SBE_SFM_SCF)
vector_output_SBW = extract_SBW(sub_bands, 200, mel_truncated_SC_SBW_SBE_SFM_SCF)
vector_output_SBE = extract_SBE(sub_bands, 200, mel_truncated_SC_SBW_SBE_SFM_SCF)
vector_output_SFM = extract_SFM(sub_bands, 200, mel_truncated_SC_SBW_SBE_SFM_SCF)
vector_output_SCF = extract_SCF(sub_bands, 200, mel_truncated_SC_SBW_SBE_SFM_SCF)
vector_output_RE = extract_RE(sub_bands, 200, mel_truncated_RE_SE, 3)
vector_output_SE = extract_SE(sub_bands, 200, mel_truncated_RE_SE)
vector_output_SC_SE = np.concatenate([vector_output_SC, vector_output_SE])
"""

def get_features(file_name_array):
    SC_features = []
    SBW_features = []
    SBE_features = []
    SFM_features = []
    SCF_features = []
    RE_features = []
    SE_features = []
    SC_SE_features = []


    for i in file_name_array:
        samples_SC_SBW_SBE_SFM_SCF, sampling_rate_SC_SBW_SBE_SFM_SCF, mel_SC_SBW_SBE_SFM_SCF, mel_truncated_SC_SBW_SBE_SFM_SCF = extract_mel_spectogram(i, 2, 200)
        samples_RE_SE, sampling_rate_RE_SE, mel_RE_SE, mel_truncated_RE_SE = extract_mel_spectogram(i, 1, 200)

        vector_output_SC = extract_SC(sub_bands, 200, mel_truncated_SC_SBW_SBE_SFM_SCF)
        SC_features.append(vector_output_SC)

        vector_output_SBW = extract_SBW(sub_bands, 200, mel_truncated_SC_SBW_SBE_SFM_SCF)
        SBW_features.append(vector_output_SBW)

        vector_output_SBE = extract_SBE(sub_bands, 200, mel_truncated_SC_SBW_SBE_SFM_SCF)
        SBE_features.append(vector_output_SBE)

        vector_output_SFM = extract_SFM(sub_bands, 200, mel_truncated_SC_SBW_SBE_SFM_SCF)
        SFM_features.append(vector_output_SFM)

        vector_output_SCF = extract_SCF(sub_bands, 200, mel_truncated_SC_SBW_SBE_SFM_SCF)
        SCF_features.append(vector_output_SCF)

        vector_output_RE = extract_RE(sub_bands, 200, mel_truncated_RE_SE, 3)
        RE_features.append(vector_output_RE)

        vector_output_SE = extract_SE(sub_bands, 200, mel_truncated_RE_SE)
        SE_features.append(vector_output_SE)

        vector_output_SC_SE = np.concatenate([vector_output_SC, vector_output_SE])
        SC_SE_features.append(vector_output_SC_SE)

    return SC_features, SBW_features, SBE_features, SFM_features, SCF_features, RE_features, SE_features, SC_SE_features



train_file_names, train_emotion_labels = get_train_name_and_label("Train")

test_file_names, test_emotion_labels = get_train_name_and_label("Test")


"""
SC_train_mel, SBW_train_mel, SBE_train_mel, SFM_train_mel, SCF_train_mel, RE_train_mel, SE_train_mel, SC_SE_train_mel = get_features(train_file_names)

SC_test_mel, SBW_test_mel, SBE_test_mel, SFM_test_mel, SCF_test_mel, RE_test_mel, SE_test_mel, SC_SE_test_mel = get_features(test_file_names)


def save_features(file_dir, filename,features):
    with open(ROOTDIR + file_dir + filename + '.spc', 'wb') as f:
        pickle.dump(features, f)

train_features = [SC_train_mel, SBW_train_mel, SBE_train_mel, SFM_train_mel, SCF_train_mel, RE_train_mel, SE_train_mel, SC_SE_train_mel]
train_features_file_name = ['SC_train_mel', 'SBW_train_mel', 'SBE_train_mel', 'SFM_train_mel', 'SCF_train_mel', 'RE_train_mel', 'SE_train_mel', 'SC_SE_train_mel']

test_features = [SC_test_mel, SBW_test_mel, SBE_test_mel, SFM_test_mel, SCF_test_mel, RE_test_mel, SE_test_mel, SC_SE_test_mel]
test_features_file_name = ['SC_test_mel', 'SBW_test_mel', 'SBE_test_mel', 'SFM_test_mel', 'SCF_test_mel', 'RE_test_mel', 'SE_test_mel', 'SC_SE_test_mel']

for i in range(len(train_features)):
    save_features('/train_features/',train_features_file_name[i], train_features[i])

for i in range(len(test_features)):
    save_features('/test_features/',test_features_file_name[i], test_features[i])
"""


scaler = StandardScaler()


#------------------------------ load training features ------------------------------
with open(ROOTDIR + '/train_features/SC_train_mel.spc', 'rb' ) as f:
    train_features_SC = pickle.load(f)
    train_features_SC = scaler.fit_transform(train_features_SC)


with open(ROOTDIR + '/train_features/SBW_train_mel.spc', 'rb' ) as f:
    train_features_SBW = pickle.load(f)
    scaler.fit(train_features_SBW)

with open(ROOTDIR + '/train_features/SBE_train_mel.spc', 'rb' ) as f:
    train_features_SBE = pickle.load(f)
    train_features_SBW = scaler.fit_transform(train_features_SBE)

with open(ROOTDIR + '/train_features/SFM_train_mel.spc', 'rb' ) as f:
    train_features_SFM = pickle.load(f)
    train_features_SFM = scaler.fit_transform(train_features_SFM)

with open(ROOTDIR + '/train_features/SCF_train_mel.spc', 'rb' ) as f:
    train_features_SCF = pickle.load(f)
    train_features_SCF = scaler.fit_transform(train_features_SCF)

with open(ROOTDIR + '/train_features/RE_train_mel.spc', 'rb' ) as f:
    train_features_RE = pickle.load(f)
    train_features_RE = scaler.fit_transform(train_features_RE)

with open(ROOTDIR + '/train_features/SE_train_mel.spc', 'rb' ) as f:
    train_features_SE = pickle.load(f)
    train_features_SE = scaler.fit_transform(train_features_SE)

with open(ROOTDIR + '/train_features/SC_SE_train_mel.spc', 'rb' ) as f:
    train_features_SC_SE = pickle.load(f)
    train_features_SC_SE = scaler.fit_transform(train_features_SC_SE)

# ------------------------------ load testing features ------------------------------
with open(ROOTDIR + '/test_features/SC_test_mel.spc', 'rb') as f:
    test_features_SC = pickle.load(f)
    test_features_SC = scaler.fit_transform(test_features_SC)


with open(ROOTDIR + '/test_features/SBW_test_mel.spc', 'rb') as f:
    test_features_SBW = pickle.load(f)
    test_features_SBW = scaler.fit_transform(test_features_SBW)

with open(ROOTDIR + '/test_features/SBE_test_mel.spc', 'rb') as f:
    test_features_SBE = pickle.load(f)
    test_features_SBE = scaler.fit_transform(test_features_SBE)

with open(ROOTDIR + '/test_features/SFM_test_mel.spc', 'rb') as f:
    test_features_SFM = pickle.load(f)
    test_features_SFM = scaler.fit_transform(test_features_SFM)

with open(ROOTDIR + '/test_features/SCF_test_mel.spc', 'rb') as f:
    test_features_SCF = pickle.load(f)
    test_features_SCF = scaler.fit_transform(test_features_SCF)

with open(ROOTDIR + '/test_features/RE_test_mel.spc', 'rb') as f:
    test_features_RE = pickle.load(f)
    test_features_RE = scaler.fit_transform(test_features_RE)

with open(ROOTDIR + '/test_features/SE_test_mel.spc', 'rb') as f:
    test_features_SE = pickle.load(f)
    test_features_SE = scaler.fit_transform(test_features_SE)



with open(ROOTDIR + '/test_features/SC_SE_test_mel.spc', 'rb') as f:
    test_features_SC_SE = pickle.load(f)
    test_features_SC_SE = scaler.fit_transform(test_features_SC_SE)



train_features_SC_SFM = np.concatenate([train_features_SC, train_features_SFM], axis=1)
print(train_features_SC_SFM.shape)

test_features_SC_SFM = np.concatenate([test_features_SC, test_features_SFM], axis=1)
print(test_features_SC_SFM.shape)



def SVM_and_AdaBoost_Classifier(train_features, train_labels, test_features, test_labels, spectral_feature_name):
    svm_classifier = SVC(kernel = 'sigmoid',C=20, random_state=10)
    svm_classifier.fit(train_features, train_labels)
    predicted_labels_SVM = svm_classifier.predict(test_features)
    #print(predicted_labels_SVM)
    cm_SVM = confusion_matrix(test_labels, predicted_labels_SVM)
    print("The confusion matrix for the SVM classifier for {} is: \n{}".format(spectral_feature_name, cm_SVM))

    acc_SVM = (accuracy_score(test_labels, predicted_labels_SVM)) * 100
    print("The accuracy score for the SVM classifier for {} is {:.3f}%\n".format(spectral_feature_name,acc_SVM))



    adaboost_classifier = AdaBoostClassifier(n_estimators = 150, random_state=10)
    adaboost_classifier.fit(train_features, train_labels)
    predicted_labels_adaboost = adaboost_classifier.predict(test_features)
    #print(predicted_labels_adaboost)
    cm_adaboost = confusion_matrix(test_labels, predicted_labels_adaboost)
    print("The confusion matrix for the AdaBoost classifier for {} is: \n{}".format(spectral_feature_name,cm_adaboost))

    acc_adaboost = (accuracy_score(test_labels, predicted_labels_adaboost)) * 100
    print("The accuracy score for the AdaBoost classifier for {} is {:.3f}%\n\n".format(spectral_feature_name,acc_adaboost))



SVM_and_AdaBoost_Classifier(train_features_SC, train_emotion_labels, test_features_SC, test_emotion_labels, 'Spectral Centroid (SC)')

SVM_and_AdaBoost_Classifier(train_features_SBW, train_emotion_labels, test_features_SBW, test_emotion_labels, 'Spectral Bandwidth (SBW)')

SVM_and_AdaBoost_Classifier(train_features_SBE, train_emotion_labels, test_features_SBE, test_emotion_labels, 'Spectral Band Energy (SBE)')

SVM_and_AdaBoost_Classifier(train_features_SFM, train_emotion_labels, test_features_SFM, test_emotion_labels, 'Spectral Flatness Measure (SFM)')

SVM_and_AdaBoost_Classifier(train_features_SCF, train_emotion_labels, test_features_SCF, test_emotion_labels, 'Spectral Crest Factor (SCF)')

SVM_and_AdaBoost_Classifier(train_features_RE, train_emotion_labels, test_features_RE, test_emotion_labels, 'Renyi Entropy (RE)')

SVM_and_AdaBoost_Classifier(train_features_SE, train_emotion_labels, test_features_SE, test_emotion_labels, 'Shannon Entropy (SE)')

SVM_and_AdaBoost_Classifier(train_features_SC_SE, train_emotion_labels, test_features_SC_SE, test_emotion_labels, 'Spectral Centroid (SC) + Shannon Entropy (SE)')

SVM_and_AdaBoost_Classifier(train_features_SC_SFM, train_emotion_labels, test_features_SC_SFM, test_emotion_labels, 'Spectral Centroid (SC) + Spectral Flatness Feature (SFM)')

"""
#------------------- randomly choose an audio file from the test set for each emotion ------------------------------
test_random_angry_audio_file = random.choice(os.listdir(os.path.join(ROOTDIR, 'EmotionSpeech', 'Test', 'Angry')))
print("test_random_angry_audio_file: {}".format(test_random_angry_audio_file))
samples_angry, sampling_rate_angry, mel_angry, mel_truncated_angry = extract_mel_spectogram(os.path.join(ROOTDIR,'EmotionSpeech/Test/Angry', test_random_angry_audio_file), 2, 200)
spectrogram(mel_angry, sampling_rate_angry)

test_random_calm_audio_file = random.choice(os.listdir(os.path.join(ROOTDIR, 'EmotionSpeech', 'Test', 'Calm')))
print("test_random_calm_audio_file: {}".format(test_random_calm_audio_file))
samples_calm, sampling_rate_calm, mel_calm, mel_truncated_calm = extract_mel_spectogram(os.path.join(ROOTDIR,'EmotionSpeech/Test/Calm', test_random_calm_audio_file), 2, 200)
spectrogram(mel_calm, sampling_rate_calm)

test_random_happy_audio_file = random.choice(os.listdir(os.path.join(ROOTDIR, 'EmotionSpeech', 'Test', 'Happy')))
print("test_random_happy_audio_file: {}".format(test_random_happy_audio_file))
samples_happy, sampling_rate_happy, mel_happy, mel_truncated_happy = extract_mel_spectogram(os.path.join(ROOTDIR,'EmotionSpeech/Test/Happy', test_random_happy_audio_file), 2, 200)
spectrogram(mel_happy, sampling_rate_happy)

test_random_sad_audio_file = random.choice(os.listdir(os.path.join(ROOTDIR, 'EmotionSpeech', 'Test', 'Sad')))
print("test_random_sad_audio_file: {}".format(test_random_sad_audio_file))
samples_sad, sampling_rate_sad, mel_sad, mel_truncated_sad = extract_mel_spectogram(os.path.join(ROOTDIR,'EmotionSpeech/Test/Sad', test_random_sad_audio_file), 2, 200)
spectrogram(mel_sad, sampling_rate_sad)
"""