import os
import librosa
import librosa.display
import numpy as np

# Base directory containing genre folders with .wav files
base_dir = "C:\\Users\\14252\\Desktop\\CSES\\genres_original"

# Output directory for spectrograms
output_dir = "C:\\Users\\14252\\Desktop\\CSES\\spectrograms_npy"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Loop through each genre folder
genre_folders = sorted(os.listdir(base_dir))
start_index = 5  # Start from the 6th folder (index 5)
for index, genre_folder in enumerate(genre_folders[:start_index], start=start_index):
    genre_path = os.path.join(base_dir, genre_folder)
    
    # Ensure it's a directory
    if not os.path.isdir(genre_path):
        continue

    print(f"Processing genre: {genre_folder}")

    # Create a corresponding folder in the output directory
    genre_output_path = os.path.join(output_dir, genre_folder)
    os.makedirs(genre_output_path, exist_ok=True)

    # Loop through each .wav file in the genre folder
    for file in sorted(os.listdir(genre_path)):
        if file.endswith(".wav"):
            file_path = os.path.join(genre_path, file)
            print(f"Processing file: {file_path}")

            # Try to load the audio file
            try:
                y, sr = librosa.load(file_path)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue  # Skip problematic file

            # Compute spectrogram
            D = librosa.stft(y)
            S_db = librosa.amplitude_to_db(abs(D), ref=np.max)

            # Save spectrogram as a NumPy array instead of an image
            output_file_path = os.path.join(genre_output_path, f"{file.replace('.wav', '.npy')}")
            np.save(output_file_path, S_db)  # Save raw spectrogram values

            print(f"Saved spectrogram as .npy: {output_file_path}")
