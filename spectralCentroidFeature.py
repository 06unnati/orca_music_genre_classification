import os
import numpy as np
import librosa
import pandas as pd

# Define directories
spectrogram_dir = r"C:\Users\14252\Desktop\CSES\trainingSpectrogramFiles"

# Initialize empty list to store data
data = []

def extract_spectral_centroid(spectrogram, sr):
    """
    Calculate spectral centroid from a magnitude spectrogram.
    """
    if spectrogram.ndim != 2:
        raise ValueError(f"Expected a 2D spectrogram, but got shape {spectrogram.shape}")

    # Convert to power spectrogram if necessary
    if np.min(spectrogram) < 0:  # If values are in log scale
        spectrogram = librosa.db_to_power(spectrogram)

    spectral_centroids = librosa.feature.spectral_centroid(S=spectrogram, sr=sr)[0]
    return np.mean(spectral_centroids)  # Store the mean value


def extract_spectral_flux(spectrogram):
    """
    Calculate spectral flux from a magnitude spectrogram.
    """
    # Calculate the difference between consecutive frames
    diff_spectrogram = np.diff(spectrogram, axis=1)

    # Calculate the squared difference
    squared_diff = np.square(np.abs(diff_spectrogram))

    # Sum the squared differences across frequency bins
    spectral_flux = np.sum(squared_diff, axis=0)

    return np.mean(spectral_flux)  # Return the mean spectral flux


# Loop through each genre folder
for genre in os.listdir(spectrogram_dir):
    genre_folder = os.path.join(spectrogram_dir, genre)
    if not os.path.isdir(genre_folder):
        continue  # Skip non-directory files

    print(f"Processing genre: {genre}")

    # Process all spectrogram .npy files in the genre folder
    for filename in os.listdir(genre_folder):
        if filename.endswith(".npy"):  # Ensure we only process npy files
            spectrogram_path = os.path.join(genre_folder, filename)

            # Load the spectrogram from the npy file
            spectrogram = np.load(spectrogram_path)

            # Ensure spectrogram has correct shape
            if spectrogram.shape[0] > spectrogram.shape[1]:  # If flipped, transpose it
                spectrogram = spectrogram.T

            # Assume standard sample rate
            sr = 44100  # CD quality

            # Compute spectral centroid
            spectral_centroid = extract_spectral_centroid(spectrogram, sr)

            # Compute spectral flux
            spectral_flux = extract_spectral_flux(spectrogram)

            # Extract file number from filename (format: genre.number.npy)
            file_number = filename.split(".")[1]  # Extract the middle part

            # Append data to list
            data.append([genre, file_number, spectral_centroid, spectral_flux])

# Convert to DataFrame
df = pd.DataFrame(data, columns=["genre", "file_number", "spectral_centroid", "spectral_flux_mean"])

# Save to CSV
df.to_csv("extracted_features.csv", index=False)

print("All spectrograms processed successfully! Data saved to spectral_centroid_data.csv")
