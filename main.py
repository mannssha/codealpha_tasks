import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

def load_data(dataset_path):
    features = []
    labels = []
    for file in os.listdir(dataset_path):
        if file.endswith(".wav"):
            emotion_label = int(file.split("-")[2])
            emotion_map = {
                1: "neutral", 2: "calm", 3: "happy", 4: "sad", 
                5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"
            }
            features.append(extract_features(os.path.join(dataset_path, file)))
            labels.append(emotion_map[emotion_label])
    return np.array(features), np.array(labels)

def preprocess_data(X, y):
    le = LabelEncoder()
    y_encoded = to_categorical(le.fit_transform(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    return X_train, X_test, y_train, y_test, le

def build_model(num_classes):
    model = Sequential()
    model.add(LSTM(128, input_shape=(40, 1), return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    dataset_path = "path_to_ravdess_dataset"  # <- Change this path
    X, y = load_data(dataset_path)
    X_train, X_test, y_train, y_test, le = preprocess_data(X, y)
    model = build_model(y_train.shape[1])
    model.summary()
    history = model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test))
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.2f}")
