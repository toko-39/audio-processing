import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, RadioButtons
import librosa
import sounddevice as sd
import math
import tkinter as tk
from tkinter import filedialog
import sys

def select_file():
    root = tk.Tk()
    root.withdraw() 
    file_path = filedialog.askopenfilename(
        title="使用するwavファイルを選択してください",
        filetypes=[("WAV files", "*.wav")]
    )
    root.destroy()
    return file_path

selected_path = select_file()
if not selected_path:
    print("ファイルが選択されませんでした。終了します。")
    sys.exit()

SR = 16000
size_frame = 512
size_shift = int(SR / 100)
db_lim = -30
spectral_envelope_number = 13
max_harmonix = 5
hamming_window = np.hamming(size_frame)

def nn2hz(notenum):
    return 440.0 * (2.0 ** ((notenum - 69) / 12.0))

def hz2nn(frequency):
    return int(round(12.0 * (math.log(frequency / 440.0) / math.log(2.0)))) + 69

def calculate_cepstrum_feature(x_frame, num_coeffs=spectral_envelope_number):
    fft_spec = np.fft.fft(x_frame)
    fft_abs_spec = np.abs(fft_spec) + 1e-6
    log_spec = np.log(fft_abs_spec)
    ceps = np.real(np.fft.fft(log_spec))
    return ceps[:num_coeffs]

def calculate_params(all_features):
    features_np = np.array(all_features)
    return np.mean(features_np, axis=0), np.var(features_np, axis=0)

def calculate_log_likelihood(feature, average, variance):
    D = len(feature)
    variance_safe = variance + 1e-9
    term1 = - (D / 2.0) * np.log(2 * np.pi)
    term2 = - 0.5 * np.sum(np.log(variance_safe))
    diff = feature - average
    term3 = - 0.5 * np.sum((diff ** 2) / variance_safe)
    return term1 + term2 + term3

def estimate_pitch_harmonic_sum(log_spec, sr, size_frame, min_nn=36, max_nn=60):
    best_nn = -1
    max_sum = -float('inf')
    delta_f = sr / size_frame
    nyquist = sr / 2
    num_bins = len(log_spec)

    for nn in range(min_nn, max_nn + 1):
        f0 = nn2hz(nn)
        h_sum = 0.0
        for h_idx in range(1, max_harmonix + 1):
            f_harmonic = h_idx * f0
            if f_harmonic > nyquist: break
            bin_index = int(round(f_harmonic / delta_f))
            if bin_index >= num_bins: break
            h_sum += log_spec[bin_index]
        
        if h_sum > max_sum:
            max_sum, best_nn = h_sum, nn
    pitch_lim = -20.0 
    return best_nn if max_sum > pitch_lim else -1

def chroma_vector(spectrum, frequencies):
    cv = np.zeros(12)
    for s, f in zip(spectrum, frequencies):
        if f <= 0:
            continue
        nn = hz2nn(f)
        cv[nn % 12] += abs(s)
    return cv

def calculate_likelihood(cv, indices, weights):
    likelihood = 0.0
    for weight, index in zip(weights, indices):
        likelihood += weight * cv[index]
    return likelihood

def train_vowel_models(filename):
    try:
        x_s, _ = librosa.load(filename, sr=SR)
    except:
        print(f"学習用ファイル {filename} が見つかりません。")
        return None
        
    vowel_segments = {
        'a': (0.5, 1.7), 'i': (1.7, 2.5), 'u': (2.5, 4.0), 'e': (4.0, 5.0), 'o': (5.0, 6.0)
    }
    models = {}
    for v, (start, end) in vowel_segments.items():
        segment = x_s[int(SR * start):int(SR * end)]
        features = []
        for i in np.arange(0, len(segment) - size_frame, size_shift):
            x_f = segment[int(i):int(i) + size_frame]
            features.append(calculate_cepstrum_feature(x_f))
        models[v] = calculate_params(features)
    return models

def separate_speech_music(y, sr):
    D = librosa.stft(y, n_fft=size_frame, hop_length=size_shift)
    Y = np.abs(D) 
    K, eps, update_times = 2, 1e-10, 100 
    F, T = Y.shape
    np.random.seed(0) #初期化
    H = np.random.rand(F, K)#特徴
    U = np.random.rand(K, T)#時間
    for i in range(update_times):
        Y_hat = np.dot(H, U)
        H = H * (np.dot(Y, U.T) / (np.dot(Y_hat, U.T) + eps))
        Y_hat = np.dot(H, U)
        U = U * (np.dot(H.T, Y) / (np.dot(H.T, Y_hat) + eps))
    Y_speech_mag = np.dot(H[:, 0:1], U[0:1, :])
    Y_music_mag = np.dot(H[:, 1:2], U[1:2, :])
    Y_total_mag = Y_speech_mag + Y_music_mag + eps
    mask_speech = Y_speech_mag / Y_total_mag
    y_speech = librosa.istft(D * mask_speech, hop_length=size_shift)
    y_music = librosa.istft(D * (1 - mask_speech), hop_length=size_shift)
    return librosa.util.fix_length(y_speech, size=len(y)), librosa.util.fix_length(y_music, size=len(y))

# 和音テンプレート作成
a_root, a_3rd, a_5th = 1.0, 0.5, 0.8
weights = [a_root, a_3rd, a_5th]
major_intervals = [0, 4, 7] 
minor_intervals = [0, 3, 7]
chroma_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
all_chord_templates = {}
chord_names_list = []
for i in range(12):
    name = chroma_labels[i] + 'maj'
    all_chord_templates[name] = [(i + interval) % 12 for interval in major_intervals]
    chord_names_list.append(name)
for i in range(12):
    name = chroma_labels[i] + 'min'
    all_chord_templates[name] = [(i + interval) % 12 for interval in minor_intervals]
    chord_names_list.append(name)

vowel_models = train_vowel_models('short.wav')

print(f"読み込み中: {selected_path}")
x_l, _ = librosa.load(selected_path, sr=SR)
x_speech, x_music = separate_speech_music(x_l, SR)

spectrogram = []
estimated_pitch_nn = []
likelihoods_results = []
estimated_chord_indices = []
vowel_list = ['a', 'i', 'u', 'e', 'o']
frequencies = np.fft.rfftfreq(size_frame, d=1.0 / SR)

for i in np.arange(0, len(x_l) - size_frame, size_shift):
    idx = int(i)
    x_frame = x_l[idx:idx + size_frame]
    current_rms = np.sqrt(np.mean(x_frame**2))
    current_db = 20 * np.log10(current_rms + 1e-6)
    fft_spec = np.fft.rfft(x_frame * hamming_window)
    mag_spec = np.abs(fft_spec)
    fft_log_abs_spec = np.log(mag_spec + 1e-6)
    spectrogram.append(fft_log_abs_spec)
    best_nn = estimate_pitch_harmonic_sum(fft_log_abs_spec, SR, size_frame)
    estimated_pitch_nn.append(best_nn if current_db >= db_lim else -1)
    
    # 母音識別
    if vowel_models:
        ceps_feat = calculate_cepstrum_feature(x_frame)
        v_likelihoods = [calculate_log_likelihood(ceps_feat, vowel_models[v][0], vowel_models[v][1]) for v in vowel_list]
        likelihoods_results.append(np.argmax(v_likelihoods))
    else:
        likelihoods_results.append(0)

    # 和音推定
    cv = chroma_vector(mag_spec, frequencies)
    max_l, best_c = -float('inf'), -1
    for name, indices in all_chord_templates.items():
        l_h = calculate_likelihood(cv, indices, weights)
        if l_h > max_l:
            max_l, best_c = l_h, chord_names_list.index(name)
    estimated_chord_indices.append(best_c if max_l > 5.0 else -1)

total_duration = (len(spectrogram) * size_shift) / SR
times = np.linspace(0, total_duration, len(spectrogram))
idx_500 = np.where(frequencies <= 500)[0]
spec_data = np.array(spectrogram).T[idx_500, :]
pitch_hz = [nn2hz(nn) if nn > 0 else np.nan for nn in estimated_pitch_nn]

# グラフ描画
fig, ax = plt.subplots(figsize=(12, 7))
plt.subplots_adjust(bottom=0.25, left=0.2)
img = ax.imshow(spec_data, aspect='auto', origin='lower', extent=[times[0], times[-1], frequencies[idx_500[0]], frequencies[idx_500[-1]]], cmap='magma', vmin=-5, vmax=5, zorder=1)
pitch_line, = ax.plot(times, pitch_hz, color='cyan', linewidth=2, label='Estimated F0', zorder=2)
v_line = ax.axvline(x=0, color='red', linestyle='--', linewidth=2, zorder=3)
vowel_text = ax.text(0.02, 0.9, '', transform=ax.transAxes, zorder=4)
chord_text = ax.text(0.02, 0.8, '', transform=ax.transAxes,  zorder=4)
ax.set_ylim(0, 500)
ax.set_title(f'NMF Separation & Vowel & Chord Analysis\nFile: {selected_path.split("/")[-1]}')
ax.set_xlabel('Time(s)')
ax.set_ylabel('Frequency (Hz)')
ax.legend(loc='upper right')

class AudioVisualizer:
    def __init__(self, full, speech, music, sr):
        self.audio_map = {"Full": full, "Speech": speech, "Music": music}
        self.current_audio, self.sr, self.ani, self.is_playing, self.stream, self.current_out_pos = full, sr, None, False, None, 0
    def set_mode(self, label): self.current_audio = self.audio_map[label]
    def _callback(self, outdata, frames, time_info, status):
        if self.is_playing:
            rem = len(self.current_audio) - self.current_out_pos
            if rem <= 0:
                outdata.fill(0)
                return
            chunk = min(frames, rem)
            outdata[:chunk, 0] = self.current_audio[self.current_out_pos : self.current_out_pos + chunk]
            if chunk < frames: outdata[chunk:, 0] = 0
            self.current_out_pos += chunk
    def update(self, frame):
        if self.is_playing:
            t = self.current_out_pos / self.sr
            if t >= total_duration:
                self.stop(None)
                return v_line, vowel_text, chord_text
            v_line.set_xdata([t])
            f_idx = int(t * SR / size_shift)
            if 0 <= f_idx < len(likelihoods_results):
                vowel_text.set_text(f"Vowel: {vowel_list[likelihoods_results[f_idx]]}")
                c_idx = estimated_chord_indices[f_idx]
                chord_text.set_text(f"Chord: {chord_names_list[c_idx] if c_idx >= 0 else 'N'}")
        return v_line, vowel_text, chord_text
    def play(self, event):
        if not self.is_playing:
            self.current_out_pos = 0
            self.stream = sd.OutputStream(samplerate=self.sr, channels=1, callback=self._callback)
            self.stream.start()
            self.is_playing = True
            self.ani.event_source.start()
    def stop(self, event):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_playing = False
        self.ani.event_source.stop()

visualizer = AudioVisualizer(x_l, x_speech, x_music, SR)
visualizer.ani = animation.FuncAnimation(fig, visualizer.update, interval=20, blit=True, cache_frame_data=False)
visualizer.ani.event_source.stop()

btn_play = Button(plt.axes([0.7, 0.05, 0.1, 0.075]), 'Play', color='lightgreen')
btn_stop = Button(plt.axes([0.82, 0.05, 0.1, 0.075]), 'Stop', color='tomato')
btn_play.on_clicked(visualizer.play)
btn_stop.on_clicked(visualizer.stop)
radio = RadioButtons(plt.axes([0.02, 0.4, 0.12, 0.2]), ('Full', 'Speech', 'Music'))
radio.on_clicked(visualizer.set_mode)

plt.show()