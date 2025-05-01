import os
import librosa
import numpy as np
import pandas as pd

# Load the existing CSV file
df = pd.read_csv('extracted_features.csv')

# Define windowing parameters
n_fft = 2048  # FFT window size
hop_length = 512  # Hop length for STFT

def extract_chroma_features(audio_path, n_fft=2048, hop_length=512):
    """
    Extract chroma vector features from an audio file.

    Args:
        audio_path (str): Path to the audio file.
        n_fft (int): FFT window size.
        hop_length (int): Hop length for STFT.

    Returns:
        numpy.ndarray: Chroma features, shape (12, t), where t is the number of time frames.
    """
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=None)  # Load audio with original sampling rate
    
    # Compute the spectrogram using STFT
    spectrogram = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))

    # Extract chroma features from the spectrogram
    chroma = librosa.feature.chroma_stft(S=spectrogram, sr=sr)

    return chroma

# Create an empty list to store the chroma features for each file
chroma_features_list = []

# Loop through each row in the existing DataFrame to process the audio files
for index, row in df.iterrows():
    genre = row['genre']
    file_number = row['file_number']
    
   # Format the file name with zero-padding (e.g., 00001, 00002, ...)
    file_name = f"{file_number:05d}"  # Pads file_number to 5 digits
    file_name = genre + "." + file_name
    
    # Construct the path to the audio file based on genre and file number
    audio_path = os.path.join(r"C:\Users\14252\Desktop\CSES\trainWavFiles", genre, f"{file_name}.wav")
    
    # Extract chroma features for the audio file
    chroma_features = extract_chroma_features(audio_path, n_fft, hop_length)

    # Calculate the mean chroma feature vector across time frames (or store the whole matrix)
    chroma_mean = np.mean(chroma_features, axis=1)  # Shape (12,) representing mean for each pitch class
    
    # Append the chroma_mean array as a new row in the list
    chroma_features_list.append(chroma_mean.tolist())  # Store as a list

    print(f"Processed {audio_path} and extracted chroma features.")

# Add the chroma feature arrays to the DataFrame in a single column
df['chroma_features'] = chroma_features_list
df.to_csv('feature_extraction.csv', index=False)

print("Feature extraction completed")
