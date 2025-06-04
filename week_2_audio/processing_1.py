import tensorflow as tf
import tensorflow_io as tfio  # Install with `pip install tensorflow-io`
import matplotlib.pyplot as plt
import sys

def load_and_convert_audio(audio_path):
    """Loads a WAV file and converts it into a spectrogram."""
    
    # Load the .wav file
    audio_binary = tf.io.read_file(audio_path)
    audio, sample_rate = tf.audio.decode_wav(audio_binary)
    
    # Convert stereo to mono if necessary
    if audio.shape[-1] > 1:
        audio = tf.reduce_mean(audio, axis=-1)  # Convert to mono
    
    # Expand dimensions (needed for batch processing)
    audio = tf.expand_dims(audio, axis=0)

    # Compute the Short-Time Fourier Transform (STFT)
    spectrogram = tf.signal.stft(audio, frame_length=256, frame_step=128)
    spectrogram = tf.abs(spectrogram)  # Get magnitude

    # Convert to a log-scale spectrogram
    spectrogram_db = tf.math.log(spectrogram + 1e-6)

    return audio, sample_rate, spectrogram_db

def plot_spectrogram(spectrogram_db, title="Spectrogram"):
    """Displays the computed spectrogram."""
    plt.figure(figsize=(10, 4))
    plt.imshow(tf.transpose(spectrogram_db)[0], aspect='auto', origin='lower')
    plt.colorbar(label='Log Magnitude')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Frequency Bin")
    plt.show()

if __name__ == "__main__":
    # Check for command-line argument
    if len(sys.argv) != 2:
        print("Usage: python process_audio.py <path_to_wav_file>")
        sys.exit(1)
    
    # Get the file path from command-line argument
    audio_path = sys.argv[1]

    # Process the audio file
    audio, sample_rate, spectrogram_db = load_and_convert_audio(audio_path)

    # Print metadata
    print(f"Processed file: {audio_path}")
    print(f"Audio shape: {audio.shape}, Sample rate: {sample_rate.numpy()}")

    # Plot the spectrogram
    plot_spectrogram(spectrogram_db, title=f"Spectrogram of {audio_path}")

