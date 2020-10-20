import numpy as np
import librosa
from pydub import AudioSegment
from pydub.utils import mediainfo
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
import os
import pickle
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

def mfcc_extraction(#.wav filename
                    audio_filename,

                    #hop_length in seconds, e.g., 0.015s (i.e., 15ms)
                    hop_duration,

                    #number of mfcc features
                    num_mfcc):

    speech = AudioSegment.from_wav(audio_filename) #Read audio data from file
    samples = speech.get_array_of_samples() #samples x(t)
    sampling_rate = speech.frame_rate #sampling rate f
    mfcc = librosa.feature.mfcc(
    np.float32(samples),
    sr = sampling_rate,
    hop_length = int(sampling_rate * hop_duration), n_mfcc = num_mfcc)

    # returns a matrix where the rows --> frames and col --> features
    return mfcc.T


def learningGMM(#list of feature vectors, each feature vector is an array
                features,

                # the number of components
                n_components,

                #maximum number of iterations
                max_iter):

    gmm = GaussianMixture(n_components = n_components, max_iter = max_iter)
    gmm.fit(features)
    return gmm


path = '/Users/johnathontoh/Desktop/Task 9.2C/Resources_9.2/SpeakerData/'
speakers = []

for files in os.listdir(path + 'Train/'):
    if files.startswith('.') is False:

        speakers.append(files)
print(speakers)



def get_name_and_label(filename):
    file_names = []
    speakers_labels = []
    for i in range(0, len(speakers)):
        sub_path = path + filename + '/' + speakers[i] + '/'
        sub_file_names = [os.path.join(sub_path, f) for f in os.listdir(sub_path)]
        sub_speakers_labels = [i] * len(sub_file_names)
        file_names += sub_file_names
        speakers_labels += sub_speakers_labels
    return file_names, speakers_labels




def get_mfcc(fileset):
    #this list is used to store the MFCC features of all training data of all speakers
    mfcc_all_speakers = []
    hop_duration = 0.015 #15ms
    num_mfcc = 12

    for s in speakers:
        sub_path = path + fileset + '/' + s + '/'
        sub_file_names = [os.path.join(sub_path, f) for f in os.listdir(sub_path)]
        # an array
        # contains the mfcc feature for 1 speaker only
        mfcc_one_speaker = np.asarray(())

        # for each file in the training data for 1 speaker
        for fn in sub_file_names:
            # mfcc features for 1 file only
            mfcc_one_file = mfcc_extraction(fn, hop_duration, num_mfcc)

            if mfcc_one_speaker.size == 0:
                mfcc_one_speaker = mfcc_one_file

            else:
                # vstack to extend the array with respect to the number of frames
                mfcc_one_speaker = np.vstack((mfcc_one_speaker, mfcc_one_file))

        mfcc_all_speakers.append(mfcc_one_speaker)
    return mfcc_all_speakers


def save_mfcc_features(foldername, mfcc_all_speakers, speakers):
    for i in range(0, len(speakers)):
        with open('/Users/johnathontoh/Desktop/Task 9.2C/Resources_9.2/' + foldername + '/' + speakers[i] + '_mfcc.fea', 'wb') as f:
            pickle.dump(mfcc_all_speakers[i], f)


def get_gmm(mfcc_all_speakers, speakers):

    n_components = 5
    max_iter = 50
    gmms = [] #list of GMMs, each is for a speaker
    for i in range(0, len(speakers)):
        gmm = learningGMM(mfcc_all_speakers[i], n_components, max_iter)
        gmms.append(gmm)
    return gmms


def save_gmms(speakers, gmms, foldername):
    for i in range(len(speakers)):
        with open('/Users/johnathontoh/Desktop/Task 9.2C/Resources_9.2/' + foldername +'/' + speakers[i] + '.gmm', 'wb') as f: #'wb' is for binary write
            pickle.dump(gmms[i], f)


def load_gmm(speakers, foldername):
    gmms = []
    for i in range(len(speakers)):
        with open('/Users/johnathontoh/Desktop/Task 9.2C/Resources_9.2/' + foldername + '/' + speakers[i] + '.gmm', 'rb') as f: #'wb' is for binary write
            gmm = pickle.load(f)
            gmms.append(gmm)
    return gmms



def speaker_recognition(audio_file_name, gmms):
    f = mfcc_extraction(audio_file_name, 0.015, 12)

    score_array = []
    # each speaker is described by each gmm
    for i in range(len(gmms)):
        score = gmms[i].score(f)
        score_array.append(score)
    max_score = max(score_array)

    speaker_id = score_array.index(max_score)
    return speaker_id, max_score


train_file_names, train_speakers_labels = get_name_and_label("Train")

test_file_names, test_speakers_labels = get_name_and_label("Test")

#---------------------- Train set ------------------------------
mfcc_all_speakers_train = get_mfcc('Train')

save_mfcc_features('TrainingFeatures',mfcc_all_speakers_train, speakers)

gmms_train = get_gmm(mfcc_all_speakers_train, speakers)

save_gmms(speakers, gmms_train, 'TrainModels')

gmm_train_models = load_gmm(speakers, 'TrainModels')



#---------------------- Test set ------------------------------
mfcc_all_speakers_test = get_mfcc('Test')

save_mfcc_features('TestingFeatures',mfcc_all_speakers_test, speakers)

gmms_test = get_gmm(mfcc_all_speakers_test, speakers)

save_gmms(speakers, gmms_test, 'TestModels')

gmm_test_models = load_gmm(speakers, 'TestModels')



speaker_id, max_score= speaker_recognition('/Users/johnathontoh/Desktop/Task 9.2C/Resources_9.2/SpeakerData/Test/Ara/a0522.wav', gmm_train_models)
print("Test speaker is 'Ara'.\n Predicted speaker is {} with a score of {}\n".format(speakers[speaker_id],max_score))

y_pred = []
for i in test_file_names:
    speaker_id, max_score = speaker_recognition(i, gmm_train_models)
    y_pred.append(speaker_id)


acc = (accuracy_score(test_speakers_labels, y_pred)) * 100
print("The accuracy score for the GMM classifier is {:.3f}%\n".format(acc))


