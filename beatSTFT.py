import os
import numpy as np
import librosa

# Define directories
audio_dir = r"C:\Users\14252\Desktop\CSES\trainWavFiles"
stft_output_dir = r"C:\Users\14252\Desktop\CSES\stftFeatures"

# Ensure the output directory exists
os.makedirs(stft_output_dir, exist_ok=True)

def compute_stft_spectrogram(y, sr):
    """
    Computes the STFT spectrogram from an audio signal.
    """
    stft = librosa.stft(y)
    magnitude_spectrogram = np.abs(stft)
    log_spectrogram = librosa.amplitude_to_db(magnitude_spectrogram, ref=np.max)
    return log_spectrogram

# Loop through each genre folder
for genre in os.listdir(audio_dir):
    genre_folder = os.path.join(audio_dir, genre)
    if not os.path.isdir(genre_folder):
        continue  # Skip non-directory files

    print(f"Processing genre: {genre}")

    # Create matching genre folder in STFT output
    genre_stft_folder = os.path.join(stft_output_dir, genre)
    os.makedirs(genre_stft_folder, exist_ok=True)

    # Process all audio files in the genre folder
    for filename in os.listdir(genre_folder):
        if filename.endswith(".wav"):  # Ensure we only process .wav files
            audio_path = os.path.join(genre_folder, filename)

            try:
                # Load the audio file
                y, sr = librosa.load(audio_path, sr=22050)  # Adjust SR if needed

                # Compute STFT spectrogram
                log_spectrogram = compute_stft_spectrogram(y, sr)

                # Save STFT spectrogram to corresponding genre folder
                stft_filename = filename.replace(".wav", "_stft.npy")
                stft_path = os.path.join(genre_stft_folder, stft_filename)
                np.save(stft_path, log_spectrogram)

                print(f"Processed {filename} -> Saved STFT to {stft_path}")

            except Exception as e:
                print(f"Skipping {filename} due to error: {e}")
                continue  # Move to the next file

print("All audio files processed successfully!")
