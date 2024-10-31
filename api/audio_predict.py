import numpy as np
from ffmpeg import FFmpeg
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

model = load_model('../saved_model/Mood_7.h5')

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

def convert_to_wav(input_file):
    ext = input_file.split('.')[-1]
    if ext == 'wav':
        return input_file
    else:
        output_path = './output/'
        output_file = f'{input_file.split(".")[-2]}.wav'.split('/')[-1]
        output = output_path+output_file
        print(input_file, output_file)
        ffmpeg = (
            FFmpeg()
            .option('y')
            .input(input_file)
            .output(output,
                vn=None,
                f="wav")
        )

        ffmpeg.execute()

        return output

def extract_features(data):
    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=SAMPLE_RATE, n_mfcc=20, n_fft=FRAME_SIZE, hop_length=(FRAME_SIZE//2)), axis=1)

    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=SAMPLE_RATE, n_mels=20, n_fft=FRAME_SIZE, hop_length=(FRAME_SIZE//2)), axis=1)

    spec_con = np.mean(librosa.feature.spectral_contrast(y=data, sr=SAMPLE_RATE, n_fft=FRAME_SIZE, hop_length=(FRAME_SIZE//2)), axis=1)

    chroma_stft = np.mean(librosa.feature.chroma_stft(y=data, sr=SAMPLE_RATE, n_fft=FRAME_SIZE, hop_length=(FRAME_SIZE//2)), axis=1)

    result = np.hstack((mfcc, mel, spec_con, chroma_stft))
    
    return result

def predict(file):
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load('./saved_model/label_classes_7.npy')

    wav_file = convert_to_wav(file)

    signal = extract_signal(wav_file)
    segment_extract = np.array(segment(signal))
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

    predict_normalization = (summed_predictions-np.min(summed_predictions))/(np.max(summed_predictions)-np.min(summed_predictions))
    pred_prob_percentage = np.multiply(predict_normalization, 100)
    pred_prob = np.vstack((label_encoder.classes_, pred_prob_percentage)).T

    return predicted_emotion, pred_prob