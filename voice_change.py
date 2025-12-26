import math
import numpy as np
import scipy.io.wavfile
import librosa
import matplotlib.pyplot as plt

SR = 16000
x,_ =librosa.load('short.wav',sr=SR)

x_a = x[int(SR * 0.5):int(SR * 1.7)]
x_i = x[int(SR * 1.7): int(SR * 2.5)]
x_u = x[int(SR * 2.5): int(SR * 4)]
x_e = x[int(SR * 4): int(SR * 5)]
x_o = x[int(SR * 5): int(SR * 6)]
    
def generate_sinusoid(sampling_rate, frequency, duration):
    sampling_interval = 1.0 / sampling_rate
    t = np.arange(sampling_rate * duration) * sampling_interval
    waveform = np.sin(2.0 * math.pi * frequency * t)
    return waveform

# 生成する正弦波の設定
frequency = 500.0
duration = len(x_i) / SR
sin_wave = generate_sinusoid(SR, frequency, duration)
sin_wave = sin_wave * 0.9

# 元の音声と正弦波を掛け合わせる
x_changed = x_i* sin_wave

x_changed_int = (x_changed * 32768.0).astype('int16')
filename = f"output_voice_change.wav"
scipy.io.wavfile.write(filename, int(SR), x_changed_int)


fft_spec = np.fft.rfft(x_i)
fft_log_abs_spec = np.log(np.abs(fft_spec) + 1e-7) 

fft_spec_changed = np.fft.rfft(x_changed)
fft_log_abs_spec_changed = np.log(np.abs(fft_spec_changed) + 1e-7)

x_data = np.linspace(0, SR/2, len(fft_log_abs_spec))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig.tight_layout(pad=5.0) 

ax1.plot(x_data, fft_log_abs_spec, color='blue')
ax1.set_title('Original Spectrum')
ax1.set_xlabel('Frequency [Hz]')
ax1.set_ylabel('Log Amplitude')
ax1.set_xlim([0, SR/2])

ax2.plot(x_data, fft_log_abs_spec_changed, color='red')
ax2.set_title(f'Changed Spectrum ({frequency}Hz)')
ax2.set_xlabel('Frequency [Hz]')
ax2.set_ylabel('Log Amplitude')
ax2.set_xlim([0, SR/2])

plt.show()