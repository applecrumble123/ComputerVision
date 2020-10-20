import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display as ipd
from pydub import AudioSegment
from pydub.utils import mediainfo
import os
from sklearn.metrics import confusion_matrix, accuracy_score


from sklearn.svm import SVC
from sklearn.ensemble  import AdaBoostClassifier

ROOTDIR = '/Users/johnathontoh/Desktop/Task 8.1P/Resources_8.1'


speech = AudioSegment.from_wav(os.path.join(ROOTDIR,'arctic_a0005.wav')) #Read audio data from file
x = speech.get_array_of_samples() #samples x(t)
x_sr = speech.frame_rate #sampling rate f - see slide 24 in week 7 lecture slides

mfcc = librosa.feature.mfcc(np.float32(x),
                            sr=x_sr,
                            # sampling rate of the signal, which is determined from the signal
                            hop_length = int(x_sr * 0.015), #15 ms
                            # number of mfcc features
                            n_mfcc = 12   )
# 12 features with 95 frames
print(mfcc.shape)

# rearrange so each row correspond to each frame with 12 features
print(mfcc.T.shape)

# ravelling the array
mfcc_flattened = np.reshape(mfcc.T, (mfcc.shape[0] * mfcc.shape[1]))

print(mfcc_flattened.shape, "\n")
plt.figure(figsize = (15, 5))
plt.plot(mfcc_flattened)
plt.ylabel('Amplitude')
#plt.show()


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

train_file_names, train_emotion_labels = get_train_name_and_label("Train")

test_file_names, test_emotion_labels = get_train_name_and_label("Test")

def mfcc_extraction(# .wav filename
                    audio_filename,

                    #hop_length in seconds, e.g., 0.015s (i.e., 15ms)
                    hop_duration,

                    #number of mfcc features
                    num_mfcc,

                    #number of frames
                    num_frames ):

    speech = AudioSegment.from_wav(audio_filename) #Read audio data from file
    samples = speech.get_array_of_samples() #samples x(t)
    sampling_rate = speech.frame_rate #sampling rate f - see slide 24 in week 7 lecture slides

    mfcc = librosa.feature.mfcc(np.float32(samples),
                                sr = sampling_rate,
                                hop_length = int(sampling_rate * hop_duration),
                                n_mfcc = num_mfcc)

    mfcc_truncated = np.zeros((num_mfcc, num_frames), np.float32)
    for i in range(min(num_frames, mfcc.shape[1])):
        mfcc_truncated[:, i] = mfcc[:, i]

    #output is a vector including mfcc_truncated.shape[0] * mfcc_truncated.shape[1] elements
    vector_output = np.reshape(mfcc_truncated.T, mfcc_truncated.shape[0] * mfcc_truncated.shape[1])

    return vector_output

num_mfcc_array =  [12, 14, 16, 18, 20, 22, 24, 26, 28, 30]

for i in num_mfcc_array:

    def get_features(file_name, num_mfcc):

        feature_vector = []

        for i in file_name:
            vector_output = mfcc_extraction(# .wav filename
                            audio_filename = i,

                            # hop_length in seconds, e.g., 0.015s (i.e., 15ms)
                            hop_duration = 0.015,

                            # number of mfcc features
                            num_mfcc = num_mfcc,

                            # number of frames
                            num_frames = 200)
            feature_vector.append(vector_output)

        return feature_vector

    train_MFCC_feature_vector = get_features(train_file_names, i)

    test_MFCC_feature_vector = get_features(test_file_names, i)

    print("------ For num_mfcc = {} ------".format(i))

    svm_classifier = SVC(random_state=10)
    svm_classifier.fit(train_MFCC_feature_vector, train_emotion_labels)
    predicted_labels_SVM = svm_classifier.predict(test_MFCC_feature_vector)
    #print(predicted_labels_SVM)
    cm_SVM = confusion_matrix(test_emotion_labels, predicted_labels_SVM)
    print("The confusion matrix for the SVM classifier is: \n{}".format(cm_SVM))

    acc_SVM = (accuracy_score(test_emotion_labels, predicted_labels_SVM)) * 100
    print("The accuracy score for the SVM classifier is {:.3f}%\n".format(acc_SVM))



    adaboost_classifier = AdaBoostClassifier(random_state=10)
    adaboost_classifier.fit(train_MFCC_feature_vector, train_emotion_labels)
    predicted_labels_adaboost = adaboost_classifier.predict(test_MFCC_feature_vector)
    #print(predicted_labels_adaboost)
    cm_adaboost = confusion_matrix(test_emotion_labels, predicted_labels_adaboost)
    print("The confusion matrix for the AdaBoost classifier is: \n{}".format(cm_adaboost))

    acc_adaboost = (accuracy_score(test_emotion_labels, predicted_labels_adaboost)) * 100
    print("The accuracy score for the AdaBoost classifier is {:.3f}%\n\n".format(acc_adaboost))
