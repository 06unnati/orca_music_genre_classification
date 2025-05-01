import os
import numpy as np
import librosa

# Define directories
spectrogram_dir = r"C:\Users\14252\Desktop\CSES\trainingSpectrogramFiles"
mfcc_output_dir = r"C:\Users\14252\Desktop\CSES\mfccFeatures"

# Ensure the output directory exists
os.makedirs(mfcc_output_dir, exist_ok=True)

def extract_mfcc_from_spectrogram(spectrogram, sr, n_mfcc=20):
    """
    Extracts MFCC features from a given spectrogram.
    """
    mel_basis = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=128)
    mel_spectrogram = np.dot(mel_basis, np.abs(spectrogram)**2)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    mfccs = librosa.feature.mfcc(S=log_mel_spectrogram, sr=sr, n_mfcc=n_mfcc)
    return mfccs

# Loop through each genre folder
for genre in os.listdir(spectrogram_dir):
    genre_folder = os.path.join(spectrogram_dir, genre)
    if not os.path.isdir(genre_folder):
        continue  # Skip non-directory files

    print(f"Processing genre: {genre}")

    # Create matching genre folder in mfcc output
    genre_mfcc_folder = os.path.join(mfcc_output_dir, genre)
    os.makedirs(genre_mfcc_folder, exist_ok=True)

    # Process all spectrogram .npy files in the genre folder
    for filename in os.listdir(genre_folder):
        if filename.endswith(".npy"):  # Ensure we only process npy files
            spectrogram_path = os.path.join(genre_folder, filename)

            # Load the spectrogram from the npy file
            spectrogram = np.load(spectrogram_path)

            # Assume a fixed sample rate (modify if needed)
            sr = 22050  # Adjust based on your dataset

            # Compute MFCCs
            mfccs = extract_mfcc_from_spectrogram(spectrogram, sr)

            # Save MFCCs to corresponding genre folder
            mfcc_filename = filename.replace(".npy", "_mfcc.npy")
            mfcc_path = os.path.join(genre_mfcc_folder, mfcc_filename)
            np.save(mfcc_path, mfccs)

            print(f"Processed {filename} -> Saved MFCCs to {mfcc_path}")

print("🎉 All spectrograms processed successfully!")
