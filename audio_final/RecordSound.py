import pyaudio
import wave
import time
import subprocess
#time.sleep(11)

CHUNK = 1024 
FORMAT = pyaudio.paInt16 #paInt8
CHANNELS = 2 
RATE = 44100 #sample rate
RECORD_SECONDS = 15
WAVE_OUTPUT_FILENAME = "output10.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK) #buffer
#subprocess.call("python emotions.py", shell=True)

#time.sleep(10)

print("* recording")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data) # 2 bytes(16 bits) per channel

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()



import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from keras import regularizers

model = Sequential()

model.add(Conv1D(256, 5,padding='same',
                 input_shape=(216,1)))
model.add(Activation('relu'))
model.add(Conv1D(128, 5,padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128, 5,padding='same',))
model.add(Activation('relu'))
#model.add(Conv1D(128, 5,padding='same',))
#model.add(Activation('relu'))
#model.add(Conv1D(128, 5,padding='same',))
#model.add(Activation('relu'))
#model.add(Dropout(0.2))
model.add(Conv1D(128, 5,padding='same',))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(16))
model.add(Activation('softmax'))
opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)

model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])

# loading json and creating model
from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("saved_models/Emotion_Voice_Detection_Model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

data, sampling_rate = librosa.load('output10.wav')
#% pylab inline
import os
import pandas as pd
import librosa
import glob 

plt.figure(figsize=(15, 5))
librosa.display.waveplot(data, sr=sampling_rate)

#livedf= pd.DataFrame(columns=['feature'])
X, sample_rate = librosa.load('./Integration/output10.wav', res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
sample_rate = np.array(sample_rate)
mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
featurelive = mfccs
livedf2 = featurelive

livedf2= pd.DataFrame(data=livedf2)
livedf2 = livedf2.stack().to_frame().T
twodim= np.expand_dims(livedf2, axis=2)
livepreds = loaded_model.predict(twodim, 
                         batch_size=32, 
                         verbose=1)
livepreds1=livepreds.argmax(axis=1)
liveabc = livepreds1.astype(int).flatten()
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
import joblib
lb=joblib.load('lb.pkl')
livepredictions = (lb.inverse_transform((liveabc)))
with open('audio.txt','w') as f:
    f.write(str(livepredictions[0].split("_")[1]))
f.close()
#print(livepredictions)

# ########################## get score ###############################################


import operator
import random
import os 

scores = {'happy' : 9, 'neutral' : 8, 'sad' : 4, 'angry' : 3, 'surprise' : 5, 'fear' : 2}

audio_comments = {'happy1' : "You have a confident, engaging and enthusiastic voice.",
                 'happy2' : "Awesome! You sounded pretty confident, keep it up!",
                 'happy3' : "Great, you have a spirited and enthusiastic voice.",
                 'neutral1' : "Well done but you could sound a bit cheerful and enthusiastic.",
                 'neutral2' : "A bit of joy in your voice would have been better.",
                 'neutral3' : "Your voice was a bit monotonous. Try to change the tone a bit",
                 'sad1' : "Your voice is a bit dull, try to sound a bit cheerful and energetic.",
                 'sad2' : "You sounded a bit sad, try to bring some excitement in your voice.",
                 'sad3' : "Sadness detected in your voice, try to sound a bit more cheerful.",
                 'angry1' : "Slight aggressiveness detected in voice, try speaking a bit calmly.", 
                 'angry2' : "Your voice might seem a bit threatening, try relaxing a bit.",
                 'angry3' : "Try to sound a bit more friendly.",
                 'fear1' : "Detected slight fear in your voice, try to sound more confident.",
                 'fear2' : "Your voice sounded slightly scared, be confident!",
                 'fear3' : "Accept your fear of speaking up and try to overcome it."}

image_comments = {'happy' : "Your expressions show you're noticeably enthusiastic, you're doing great!",
                'neutral' : "", 
                'sad' : "Try having brighter facial expressions.",
                'angry' : "Angry emotions were observed at times.",
                'fear' : "At some moments, you looked a bit scared. Try practicing a few more times.",
                'surprise' : "There were few points where you looked somewhat surprised.",
                'happy_low' : "Try smiling more frequently.",
                'neutral_low' : "A smile would have been great.",
                'sad_low' : "You looked a bit upset to us, try to keep positive or neutral expressions.",
                'angry_low' : "A smiling face would be a great plus.",
                'fear_low' : "We sensed a little fear from your expressions, try to work on that.",
                'surprise_low' : "Try to keep more neutral/happy facial expressions." }


hr_comemnts = { 'high' : "we also sensed that your hearbeat was a bit high, it is common, but try to take a deep breath and chill",
                'normal': "Your hearbeat was normal, That's great!"}

audio = open("audio.txt","r")
image = open("image.txt","r")

image_result = {}
with image as f:
    for line in f:
        res = line.split(' ')
        image_result[res[0]]=int(res[1].rstrip())

image_result = sorted(image_result.items(), key=operator.itemgetter(1), reverse=True)
image_major = image_result[0][0]
audio_major = audio.read()

if(audio_major == 'calm'):
    audio_major = 'neutral'

if(audio_major == 'disgust'):
    audio_major = 'sad'

if(audio_major == 'fearful'):
    audio_major = 'fear'

if(audio_major == 'surprised'):
    audio_major = 'surprise'

score = (1.5*scores[audio_major] + scores[image_major])/2.5
audio_major = audio_major + str(random.randint(1,3))

comments = ""
# print image_result

for i in range(len(image_result)):
    key = image_result[i][0]
    value = image_result[i][1]
    if(value >= 30):
        comments += " " +image_comments[key]
    elif(value >= 5):
        comments += " " +image_comments[key+'_low']

comments = comments + " " + audio_comments[audio_major]

score = score + random.randint(-5,5)/10
score_to_print = str(score)+"/10"
file = open("result.txt",'w')
file.write(score_to_print+":"+comments)
file.close()

os.system("adb push result.txt /sdcard/CSE/")
os.system("adb shell am start com.qualcomm.hercoders.cse/.MainActivity")




















