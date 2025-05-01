import os
import numpy as np
import pandas as pd
from scipy.fft import fft
import librosa

# Define directories
spectrogram_dir = r"C:\Users\14252\Desktop\CSES\trainingSpectrogramFiles"

# Load the existing CSV file
df = pd.read_csv('feature_extraction.csv')

def calculate_hnr(audio_data, sampling_rate, frame_size=0.05, overlap=0.5, epsilon=1e-10):
    """
    Calculates the Harmonic-to-Noise Ratio (HNR) of an audio signal.

    Args:
        audio_data (np.ndarray): The audio signal as a 1D numpy array.
        sampling_rate (int): The sampling rate of the audio signal in Hz.
        frame_size (float): The size of the analysis frame in seconds.
        overlap (float): The overlap between analysis frames as a fraction of the frame size.
        epsilon (float): Small value to avoid division by zero.

    Returns:
        np.ndarray: An array of HNR values for each frame.
    """
    frame_length = int(frame_size * sampling_rate)
    hop_length = int(frame_length * (1 - overlap))
    num_frames = (len(audio_data) - frame_length) // hop_length + 1
    hnr_values = []

    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frame = audio_data[start:end]

        # Calculate the FFT of the frame
        spectrum = fft(frame)
        magnitude_spectrum = np.abs(spectrum)

        # Find the fundamental frequency (peak in the spectrum)
        peak_index = np.argmax(magnitude_spectrum[1:frame_length//2+1]) + 1
        fundamental_frequency = peak_index * sampling_rate / frame_length

        # Calculate the power in the harmonic and noise components
        harmonic_range = int(0.1 * sampling_rate / fundamental_frequency)
        harmonic_power = np.sum(magnitude_spectrum[peak_index - harmonic_range : peak_index + harmonic_range+1]**2)
        total_power = np.sum(magnitude_spectrum[1:frame_length//2+1]**2)

        # Avoid division by zero
        noise_power = total_power - harmonic_power
        hnr = 10 * np.log10((harmonic_power + epsilon) / (noise_power + epsilon))  # Add epsilon to avoid division by zero

        hnr_values.append(hnr)

    return np.array(hnr_values)

# Prepare a list for storing HNR values
hnr_list = []

# Loop through each row in the CSV and process the audio files
for index, row in df.iterrows():
    genre = row['genre']
    file_number = int(row['file_number'])  # Convert to int

    # Construct file path for the original audio file
    filename = f"{genre}.{file_number:05d}.wav"  # Assuming .wav format
    audio_path = os.path.join(r"C:\Users\14252\Desktop\CSES\trainWavFiles", genre, f"{filename}")

    if os.path.exists(audio_path):
        # Load the audio file
        audio_data, sr = librosa.load(audio_path, sr=44100)

        # Compute Harmonic-to-Noise Ratio (HNR)
        hnr_values = calculate_hnr(audio_data, sr)

        # Compute the average HNR over all frames (or you could save all frame HNRs depending on the requirement)
        average_hnr = np.mean(hnr_values)

        # Append the average HNR to the list
        hnr_list.append(average_hnr)
    else:
        print(f"Warning: File not found {audio_path}")
        hnr_list.append(np.nan)  # Append NaN for missing files, to ensure proper CSV formatting

# Add the new HNR values to the DataFrame
df['hnr'] = hnr_list

# Check if the file exists and remove it if so
if os.path.exists('feature_extraction.csv'):
    os.remove('feature_extraction.csv')

# Save the updated DataFrame to the CSV file
df.to_csv('feature_extraction.csv', index=False)

print("Feature extraction completed and HNR added to 'feature_extraction.csv'")
