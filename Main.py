import pandas as pd

# 1.1 Read the CSV
df = pd.read_csv('BirdsVoice.csv')

# 1.2 Compute total recordings per species
counts = df['common_name'].value_counts()
top20 = counts.nlargest(20).index

# 1.3 Filter to top 20 and reset index
df = df[df['common_name'].isin(top20)].reset_index(drop=True)
print(df.shape)            # ~600 × 10
print(df['common_name'].unique())

import os

# 2.1 Convert “M:SS” → seconds
def length_to_seconds(x):
    m, s = x.split(':')
    return int(m) * 60 + int(s)

df['duration_s'] = df['recording_length'].apply(length_to_seconds)

# 2.2 Assume you’ve downloaded audio into ./audio/, named by xc_id:
df['filepath'] = df['xc_id'].apply(lambda id: os.path.join('audio', f'{id}.mp3'))

# 2.3 Quick check
print(df[['filepath','duration_s']].head())

from sklearn.model_selection import train_test_split

# 3.1 Label encoding
label_map = {name:idx for idx,name in enumerate(top20)}
df['label'] = df['common_name'].map(label_map)

# 3.2 60/20/20 stratified split
trainval, test = train_test_split(df, test_size=0.20, 
                                  stratify=df['label'], random_state=42)
train, val   = train_test_split(trainval, test_size=0.25,  # 0.25×0.8 = 0.20
                                 stratify=trainval['label'], random_state=42)

print(train.shape, val.shape, test.shape)

import numpy as np
import librosa
import os

# Audio settings
SR = 22050  # sampling rate
DURATION = 5.0  # seconds to load/pad
N_MELS = 128  # number of mel bands

def extract_mel(path):
    """
    Load an audio file, pad or truncate to exactly DURATION seconds,
    compute a N_MELS-band mel-spectrogram, convert to dB, and z-score normalize.
    Returns:
        m_norm: np.ndarray of shape (N_MELS, T) or None if loading fails
    """
    try:
        # Check if file exists
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return None
            
        # 1) Load (will truncate if longer than DURATION)
        y, _ = librosa.load(path, sr=SR, mono=True, duration=DURATION)
        
        # Check if audio was loaded successfully
        if len(y) == 0:
            print(f"Empty audio file: {path}")
            return None
        
        # 2) Pad with zeros if too short
        target_len = int(SR * DURATION)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        
        # 3) Compute mel-spectrogram
        m = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS)
        m_db = librosa.power_to_db(m, ref=np.max)
        
        # 4) Z-score normalization
        m_norm = (m_db - m_db.mean()) / (m_db.std() + 1e-6)
        
        return m_norm
        
    except Exception as e:
        print(f"Error processing {path}: {str(e)}")
        return None

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Create dummy data that matches your expected shapes
# This function generates random mel spectrograms and labels to simulate this dataset and ensure it works.
def create_dummy_dataset(n_samples=600, n_classes=20, n_mels=128, time_steps=216):
    """
    Create dummy audio data for testing your model
    """
    # Generate random mel spectrograms
    X = np.random.randn(n_samples, n_mels, time_steps, 1).astype(np.float32)
    
    # Generate random labels
    y_labels = np.random.randint(0, n_classes, n_samples)
    y = tf.keras.utils.to_categorical(y_labels, num_classes=n_classes)
    
    return X, y

# Replace your data loading section with this:
print("Creating dummy dataset for testing...")

# Create dummy data
X_dummy, y_dummy = create_dummy_dataset(n_samples=600, n_classes=20)

# Split into train/val/test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_dummy, y_dummy, test_size=0.2, random_state=42, stratify=y_dummy.argmax(axis=1)
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, 
    stratify=y_train_val.argmax(axis=1)
)

# Define input shape and label map
input_shape = X_train.shape[1:]  # (128, 216, 1)
label_map = {f"species_{i}": i for i in range(20)}

print("Dataset created successfully!")
print("Input shape:", input_shape)
print("Train/Val/Test sizes:", X_train.shape[0], X_val.shape[0], X_test.shape[0])

# Now your model code will work:
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(label_map), activation='softmax'),
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=32,
    epochs=5,  # Reduced for testing
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
)

print("Model training completed!")

import os
import glob
import numpy as np
import tensorflow as tf

# Helper to find the single .mp3 under audio/ matching an xc_id
def find_audio_path(xc_id):
    # recursive glob for e.g. audio/**/XC66646.mp3
    pattern = os.path.join('audio', '**', f'{xc_id}.mp3')
    matches = glob.glob(pattern, recursive=True)
    return matches[0] if matches else None

# 5.0 — rebuild df['filepath'] using the finder
df['filepath'] = df['xc_id'].apply(find_audio_path)

# 5.1 — report & drop missing
missing = df['filepath'].isna()
if missing.any():
    print(f"Warning: {missing.sum()} files not found, dropping those rows:")
    print(df.loc[missing, 'xc_id'].tolist())
    df = df[~missing].reset_index(drop=True)

# 5.2 — build_dataset now only sees real files and handles no file cases
def build_dataset(df_subset):
    X, y = [], []
    failed_files = []
    
    for _, row in df_subset.iterrows():  # Fixed: removed asterisks
        feat = extract_mel(row['filepath'])
        # Skip if extraction failed (feat is None)
        if feat is None:
            failed_files.append(row['filepath'])
            continue
        X.append(feat[..., np.newaxis])  # → (128, T, 1)
        y.append(row['label'])
    
    if failed_files:
        print(f"Skipped {len(failed_files)} files that couldn't be loaded")
    
    if len(X) == 0:
        raise ValueError("No valid audio files found! Please check your file paths and audio files.")
    
    X = np.stack(X, axis=0)  # → (N, 128, T, 1)
    y = tf.keras.utils.to_categorical(
        y,
        num_classes=len(label_map)
    )
    return X, y

# 5.3 — apply to splits
X_train, y_train = build_dataset(train)
X_val,   y_val   = build_dataset(val)
X_test,  y_test  = build_dataset(test)

# 5.4 — sanity check
input_shape = X_train.shape[1:]
print("Input shape:", input_shape)
print("Sizes (train/val/test):", X_train.shape[0],
      X_val.shape[0], X_test.shape[0])


from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(label_map), activation='softmax'),
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=32,
    epochs=30,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
)


