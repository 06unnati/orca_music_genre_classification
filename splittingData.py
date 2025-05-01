import os
import shutil
from sklearn.model_selection import train_test_split

base_dir = r"C:\Users\14252\Desktop\CSES\genres_original"
spectrogram_dir = r"C:\Users\14252\Desktop\CSES\spectrograms_npy"
train_wav_dir = r"C:\Users\14252\Desktop\CSES\trainWavFiles"
test_wav_dir = r"C:\Users\14252\Desktop\CSES\testWavFiles"
train_spectrogram_dir = r"C:\Users\14252\Desktop\CSES\trainingSpectrogramFiles"
test_spectrogram_dir = r"C:\Users\14252\Desktop\CSES\testSpectrogramFiles"

def copy_files(file_list, src_folder, dest_root, genre):
    genre_dest_folder = os.path.join(dest_root, genre)
    os.makedirs(genre_dest_folder, exist_ok=True)
    for file in file_list:
        src = os.path.join(src_folder, file)
        dest = os.path.join(genre_dest_folder, file)
        shutil.copy2(src, dest)
    print(f"Copied {len(file_list)} files to {genre_dest_folder}")

for entry in os.scandir(base_dir):
    if not entry.is_dir():
        continue

    genre_folder = entry.name
    genre_path = entry.path
    genre_spectrogram_folder = os.path.join(spectrogram_dir, genre_folder)

    print(f"Processing genre: {genre_folder}")

    # Ensure spectrogram directory exists before listing files
    if os.path.exists(genre_spectrogram_folder):
        spectrogram_files = [f for f in os.listdir(genre_spectrogram_folder) if f.endswith(".npy")]
    else:
        spectrogram_files = []

    wav_files = [f for f in os.listdir(genre_path) if f.endswith(".wav")]

    # Skip this genre if no spectrograms exist
    if len(spectrogram_files) == 0:
        print(f"No spectrogram files found for {genre_folder}. Skipping...")
        continue  

    train_wav_files, test_wav_files = train_test_split(wav_files, test_size=0.2, random_state=42)
    train_spectrogram_files, test_spectrogram_files = train_test_split(spectrogram_files, test_size=0.2, random_state=42)

    copy_files(train_wav_files, genre_path, train_wav_dir, genre_folder)
    copy_files(test_wav_files, genre_path, test_wav_dir, genre_folder)
    copy_files(train_spectrogram_files, genre_spectrogram_folder, train_spectrogram_dir, genre_folder)
    copy_files(test_spectrogram_files, genre_spectrogram_folder, test_spectrogram_dir, genre_folder)

    print(f"Finished processing {genre_folder}")

print("🎉 All files successfully split while keeping genre folders!")
