import numpy as np
import wave
import struct

def generate_beep(filename, frequencies, duration_sec, is_error=False, sample_rate=44100):
    num_samples = int(sample_rate * duration_sec)
    audio = np.zeros(num_samples)
    
    # Mix frequencies
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    for freq in frequencies:
        if is_error:
            # Sawtooth-like for buzz sound
            audio += 0.5 * (2 * (t * freq - np.floor(t * freq + 0.5)))
        else:
            # Sine wave for clean tone
            audio += 0.5 * np.sin(2 * np.pi * freq * t)
            
    # Normalize
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
        
    # Scale to 16-bit
    audio = (audio * 32767).astype(np.int16)
    
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())

# Generate "Success" Beep (Double high-pitch tone)
# Simulating a double beep by silencing the middle 10%
s_freq = [880.0, 1046.5]  # A5, C6
generate_beep('success.wav', s_freq, 0.4)

# Generate "Already Marked" Beep (Neutral dull tone)
a_freq = [440.0]  # A4
generate_beep('already.wav', a_freq, 0.3)

# Generate "Error/Unrecognized" Beep (Low buzzer)
e_freq = [150.0, 200.0]
generate_beep('error.wav', e_freq, 0.5, is_error=True)

print("Generated Audio Files")
