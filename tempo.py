import os
import librosa
import numpy as np
import pandas as pd

# Define directories
spectrogram_dir = r"C:\Users\14252\Desktop\CSES\trainingSpectrogramFiles"

# Load the existing CSV file
df = pd.read_csv('feature_extraction.csv')

def extract_tempo_from_spectrogram(spectrogram, sr, hop_length):
    """
    Extracts tempo (BPM) from a spectrogram.

    Args:
        spectrogram (np.ndarray): Spectrogram of the audio.
        sr (int): Sampling rate of the audio.
        hop_length (int): Hop length used in spectrogram calculation.

    Returns:
        float: Estimated tempo in BPM.
    """
    onset_env = librosa.onset.onset_strength(sr=sr, S=spectrogram)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop_length)[0]
    return tempo

# Prepare a list for storing tempo values
tempo_list = []

# Loop through each row in the CSV and process the spectrogram files
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
        hop_length = 512  # Hop length used for STFT

        # Compute tempo (BPM)
        tempo = extract_tempo_from_spectrogram(spectrogram, sr, hop_length)

        # Append tempo to the list
        tempo_list.append(tempo)
    else:
        print(f"Warning: File not found {spectrogram_path}")
        tempo_list.append(None)  # Append None for missing files

# Add the new tempo column to the DataFrame
df['tempo'] = tempo_list

# Save the updated DataFrame back to the CSV file
output_file = 'feature_extraction.csv'

# Check if the file exists and remove it if so
if os.path.exists(output_file):
    os.remove(output_file)

# Handle saving the file, ensuring no permission issues
try:
    df.to_csv(output_file, index=False)
    print("Feature extraction completed and tempo added to 'feature_extraction.csv'")
except PermissionError:
    print(f"Permission denied: Unable to save {output_file}. Close any open instances and retry.")
