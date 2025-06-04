import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models

# Define the dataset path
DATASET_PATH = "/root/src/data/audio"


# Function to extract MFCC features
def extract_features(file_path, max_pad_len=128):
    try:
        audio, sample_rate = librosa.load(file_path, sr=22050)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

        # Pad or truncate to fixed length
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]

        return mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Load dataset
def load_data(dataset_path):
    features, labels = [], []
    
    for speaker_dir in os.listdir(dataset_path):
        speaker_path = os.path.join(dataset_path, speaker_dir)
        
        if os.path.isdir(speaker_path):  # Ensure it's a directory
            for file in os.listdir(speaker_path):
                if file.endswith(".wav"):
                    file_path = os.path.join(speaker_path, file)
                    emotion_label = file.replace(".wav", "")  # Extract label from filename
                    mfcc = extract_features(file_path)
                    
                    if mfcc is not None:
                        features.append(mfcc)
                        labels.append(emotion_label)
    
    return np.array(features), np.array(labels)

# Load and process data
X, y = load_data(DATASET_PATH)

# Review data
print(f"Feature array shape: {X.shape}")
print(f"Labels array shape: {y.shape}")
print(f"Feature data type: {X.dtype}")
print(f"Labels data type: {y.dtype}")

# Check a few label values
print("First 10 labels:", y[:10])

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Expand dimensions for CNN input
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Define CNN model
def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
#        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Model setup
input_shape = (40, 128, 1)
num_classes = len(np.unique(y))
model = build_model(input_shape, num_classes)

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Train model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Save model
#model.save("speech_emotion_recognition.h5")
model.save('my_model.keras')
