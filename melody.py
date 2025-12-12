import math
import matplotlib.pyplot as plt
import numpy as np
import librosa

SR= 16000
x_l,_ =librosa.load('kimi.wav',sr=SR)

size_frame= 512 

hamming_window = np.hamming(size_frame)

size_shift= int(16000/100) #0.01秒(10 msec)

def nn2hz(notenum):
    return 440.0 * (2.0 ** ((notenum-69) / 12.0))

def hz2nn(frequency):
    return int(round (12.0 * (math.log(frequency/440.0)/math.log(2.0))))+69

def chroma_vector(spectrum, frequencies):
    cv = np.zeros(12)

    for s, f in zip(spectrum, frequencies):
        if f <= 0:
            continue
        nn = hz2nn(f)
        cv[nn % 12] += abs(s)

    return cv

def estimate_pitch_harmonic_sum(log_spec, SR, size_frame, min_nn=36, max_nn=60):
    best_nn = -1
    max_sum = -float('inf')
    delta_f = SR / size_frame 
    nyquist = SR / 2
    num_bins = len(log_spec)
    
    max_harmonics = 5
    
    for nn in range(min_nn, max_nn + 1):
        f0 = nn2hz(nn)
        harmonic_sum = 0.0
        
        h_idx = 1
        while True:
            f_harmonic = h_idx * f0
            
            # Nyquist周波数を超えるか、10次を超える場合にブレイク
            if f_harmonic > nyquist or h_idx > max_harmonics:
                break
            
            # 対応するFFTビンインデックスを計算
            bin_index = int(round(f_harmonic / delta_f))
            
            if bin_index >= num_bins:
                break 
            
            # 対数振幅を加算
            harmonic_sum += log_spec[bin_index]
            
            h_idx += 1
            
        if harmonic_sum > max_sum:
            max_sum = harmonic_sum
            best_nn = nn
            
    pitch_lim = -20.0 
    return best_nn if max_sum > pitch_lim else -1

spectrogram=[]
chromagram = []
estimated_pitch_nn = []

frequencies = np.fft.rfftfreq(size_frame, d=1.0 / SR)

dblim = -30

for i in np.arange(0, len(x_l)-size_frame,size_shift):

    idx= int(i)
    x_frame=x_l[idx:idx+size_frame]
    
    current_rms = np.sqrt(np.mean(x_frame**2))
    current_db = 20 * np.log10(current_rms + 1e-6) 

    fft_spec = np.fft.rfft(x_frame * hamming_window)
    fft_log_abs_spec=np.log(np.abs(fft_spec) + 1e-6) 

    spectrogram.append(fft_log_abs_spec)
    
    best_nn = estimate_pitch_harmonic_sum(fft_log_abs_spec, SR, size_frame)
    
    if current_db < dblim:
         best_nn = -1 
         
    estimated_pitch_nn.append(best_nn)


estimated_pitch_hz = [nn2hz(nn) if nn != -1 else 0 for nn in estimated_pitch_nn]
spectrogram_array = np.array(spectrogram).T
duration = len(x_l) / SR 

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
time_points_pitch = np.linspace(0, duration, len(estimated_pitch_hz)) # ピッチの時間軸
time_points_wave = np.linspace(0, duration, len(x_l)) # 波形の時間軸
min_nn, max_nn = 36, 60

ax1 = axes[0]
ax1.set_title('Time Domain Signal (Waveform)')
ax1.set_ylabel('Amplitude')
ax1.plot(time_points_wave, x_l, color='gray', linewidth=0.5)
ax1.set_ylim(min(x_l) * 1.1, max(x_l) * 1.1)


ax2 = axes[1]
ax2.set_title('Spectrogram')
ax2.set_ylabel('Frequency [Hz]')
im2 = ax2.imshow(
    np.flipud(spectrogram_array),
    extent=[0, duration, 0, SR / 2],
    aspect='auto',
    interpolation='nearest',
    cmap='viridis'
)
ax2.set_ylim(0, 300) 


ax3 = axes[2]
ax3.set_title(f'Estimated Fundamental Frequency')
ax3.set_xlabel('Time [sec]')
ax3.set_ylabel('Frequency [Hz]')

ax3.plot(time_points_pitch, estimated_pitch_hz, drawstyle='steps-post', color='blue', linewidth=1.5)

y_ticks_hz = np.arange(0, 301, 50) 
y_tick_labels = [f'{int(h)} Hz' for h in y_ticks_hz]

ax3.set_ylim(0, 300) 
ax3.set_yticks(y_ticks_hz)
ax3.set_yticklabels(y_tick_labels, fontsize=8)
ax3.grid(True, axis='y', alpha=0.5)


# プロットの表示
plt.tight_layout()
plt.show()