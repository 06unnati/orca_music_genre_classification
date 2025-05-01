import os
import librosa
import numpy as np
import pandas as pd

# Define directories
spectrogram_dir = r"C:\Users\14252\Desktop\CSES\trainingSpectrogramFiles"

# Load the existing CSV file
df = pd.read_csv('feature_extraction.csv')

# Check if 'file_number' column exists
if 'file_number' not in df.columns:
    print("Error: 'file_number' column not found in extracted_features.csv. Check column names.")
    print("Existing columns:", df.columns)
    exit(1)  # Exit script if column is missing

def extract_tonal_centroid(spectrogram: np.ndarray, sr: int) -> np.ndarray:
    """
    Extracts the tonal centroid feature (Tonnetz) from a spectrogram.

    Args:
        spectrogram (np.ndarray): The spectrogram of the audio signal.
        sr (int): The sampling rate of the audio signal.

    Returns:
        np.ndarray: The tonal centroid feature (Tonnetz).
    """
    chromagram = librosa.feature.chroma_stft(S=np.abs(spectrogram), sr=sr)
    tonnetz = librosa.feature.tonnetz(chroma=chromagram, sr=sr)
    return tonnetz

# Prepare a new column for tonal centroids
tonal_centroid_list = []

# Loop through each row in the CSV
for index, row in df.iterrows():
    genre = row['genre']
    file_number = int(row['file_number'])  # Convert to int

    # Construct file path
    filename = f"{genre}.{file_number:05d}.npy"  # Example: blues.00000.npy
    spectrogram_path = os.path.join(spectrogram_dir, genre, filename)

    if os.path.exists(spectrogram_path):
        # Load the spectrogram
        spectrogram = np.load(spectrogram_path)

        # Ensure spectrogram has correct shape
        if spectrogram.shape[0] > spectrogram.shape[1]:  # If flipped, transpose it
            spectrogram = spectrogram.T

        # Assume standard sample rate
        sr = 44100  # CD quality

        # Compute tonal centroid
        tonal_centroid = extract_tonal_centroid(spectrogram, sr)

        # Append tonal centroid to list
        tonal_centroid_list.append(tonal_centroid)
    else:
        print(f"Warning: File not found {spectrogram_path}")
        tonal_centroid_list.append(None)  # Append None for missing files

# Add new feature column to the DataFrame
df['tonal_centroid'] = tonal_centroid_list

# Delete file if it exists before saving
output_file = 'feature_extraction.csv'
if os.path.exists(output_file):
    os.remove(output_file)

# Handle Windows file permissions
try:
    df.to_csv(output_file, index=False)
    print("Feature extraction completed and saved to feature_extraction.csv")
except PermissionError:
    print(f"Permission denied: Unable to save {output_file}. Close any open instances and retry.")
