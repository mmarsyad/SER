import numpy as np
import pandas as pd
import soundfile as sf
from pydub import AudioSegment
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

model = load_model('./saved_model/Mood_7.h5')
SAMPLE_RATE = 22050
FRAME_SIZE = 1024
DURATION = int(4 * SAMPLE_RATE)
HOP_SIZE = int(DURATION * 0.5)

def extract_signal(path):
    signal, _ = librosa.load(path, sr=SAMPLE_RATE)
    return signal

def segment(signal):
    frame_size = DURATION
    hop_length = HOP_SIZE
    total_samples = len(signal)

    segments = []

    for start in range(0, total_samples, hop_length):
        end = start + frame_size
        if end > total_samples:
            segment = np.pad(signal[start:total_samples], (0, end - total_samples), 'constant')
        else:
            segment = signal[start:end]
        segments.append(segment)
    
    return segments

def convert_to_wav(input_file, output_file):
    file_ext = input_file.split('.')[-1].lower()

    if file_ext == 'mp3':
        audio = AudioSegment.from_mp3(input_file)
        audio.export(output_file, format='wav')
    elif file_ext == 'opus':
        data, sr = sf.read(input_file)
        sf.write(output_file, data, sr, format='wav')
    elif file_ext == 'wav':
        pass
    else:
        raise ValueError('Format file tidak didukung.')

input_file = 'sedih.opus'
output_file = 'output.wav'
convert_to_wav(input_file, output_file)

def extract_features(data):
    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=SAMPLE_RATE, n_mfcc=20, n_fft=FRAME_SIZE, hop_length=(FRAME_SIZE//2)), axis=1)

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=SAMPLE_RATE, n_mels=20, n_fft=FRAME_SIZE, hop_length=(FRAME_SIZE//2)), axis=1)

    # Spectral Contrast
    spec_con = np.mean(librosa.feature.spectral_contrast(y=data, sr=SAMPLE_RATE, n_fft=FRAME_SIZE, hop_length=(FRAME_SIZE//2)), axis=1)

    # Chroma
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=data, sr=SAMPLE_RATE, n_fft=FRAME_SIZE, hop_length=(FRAME_SIZE//2)), axis=1)
    result = np.hstack((mfcc, mel, spec_con, chroma_stft))
    
    return result

label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('./saved_model/label_classes_7.npy')

a = extract_signal(output_file)
segment_extract = np.array(segment(a))
features = []
if len(segment_extract > 1):
    for i, signal in enumerate(segment_extract):
        features.append(extract_features(signal))
else:
    features.append(extract_features(segment_extract))
features = np.array(features)

prediction = model.predict(features)
summed_predictions = np.sum(prediction, axis=0)
predicted_label = np.argmax(summed_predictions)
predicted_emotion = label_encoder.inverse_transform([predicted_label])

print(f'This sound labeled as {predicted_emotion[0]}')