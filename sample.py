import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, RadioButtons
import librosa
import sounddevice as sd
import math

SR = 16000
SIZE_FRAME = 512
SIZE_SHIFT = int(SR / 100)
DB_LIM = -30
SPECTRAL_ENVELOPE_NUMBER = 13
MAX_HARMONICS = 5
HAMMING_WINDOW = np.hamming(SIZE_FRAME)

def nn2hz(notenum):
    return 440.0 * (2.0 ** ((notenum - 69) / 12.0))

# 周波数からノートナンバーへ
def hz2nn(frequency):
    if frequency <= 0:
        return 0
    return int(round(12.0 * (math.log(frequency / 440.0) / math.log(2.0)))) + 69

def calculate_cepstrum_feature(x_frame, num_coeffs=SPECTRAL_ENVELOPE_NUMBER):
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
        for h_idx in range(1, MAX_HARMONICS + 1):
            f_harmonic = h_idx * f0
            if f_harmonic > nyquist: break
            bin_index = int(round(f_harmonic / delta_f))
            if bin_index >= num_bins: break
            h_sum += log_spec[bin_index]
        
        if h_sum > max_sum:
            max_sum, best_nn = h_sum, nn

    return best_nn if max_sum > -20.0 else -1

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
        # クロマベクトルcvの要素を重み付けして合計
        likelihood += weight * cv[index]
    return likelihood

def train_vowel_models(filename):
    x_s, _ = librosa.load(filename, sr=SR)
    vowel_segments = {
        'a': (0.5, 1.7), 'i': (1.7, 2.5), 'u': (2.5, 4.0), 'e': (4.0, 5.0), 'o': (5.0, 6.0)
    }
    models = {}
    for v, (start, end) in vowel_segments.items():
        segment = x_s[int(SR * start):int(SR * end)]
        features = []
        for i in np.arange(0, len(segment) - SIZE_FRAME, SIZE_SHIFT):
            x_f = segment[int(i):int(i) + SIZE_FRAME]
            features.append(calculate_cepstrum_feature(x_f))
        models[v] = calculate_params(features)
    return models

def separate_speech_music(y, sr):
    D = librosa.stft(y, n_fft=SIZE_FRAME, hop_length=SIZE_SHIFT)
    Y = np.abs(D) 
    
    K = 2
    eps = 1e-10
    update_times = 100 
    
    F, T = Y.shape
    np.random.seed(0)
    H = np.random.rand(F, K)
    U = np.random.rand(K, T)
    
    for i in range(update_times):
        Y_hat = np.dot(H, U)
        H = H * (np.dot(Y, U.T) / (np.dot(Y_hat, U.T) + eps))
        Y_hat = np.dot(H, U)
        U = U * (np.dot(H.T, Y) / (np.dot(H.T, Y_hat) + eps))

    Y_speech_mag = np.dot(H[:, 0:1], U[0:1, :])
    Y_music_mag = np.dot(H[:, 1:2], U[1:2, :])
    
    Y_total_mag = Y_speech_mag + Y_music_mag + eps
    mask_speech = Y_speech_mag / Y_total_mag
    
    y_speech = librosa.istft(D * mask_speech, hop_length=SIZE_SHIFT)
    y_music = librosa.istft(D * (1 - mask_speech), hop_length=SIZE_SHIFT)
    
    y_speech = librosa.util.fix_length(y_speech, size=len(y))
    y_music = librosa.util.fix_length(y_music, size=len(y))
    
    return y_speech, y_music

# 和音テンプレート作成
a_root, a_3rd, a_5th = 1.0, 0.5, 0.8
weights = [a_root, a_3rd, a_5th]
major_intervals = [0, 4, 7] 
minor_intervals = [0, 3, 7]
chroma_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

all_chord_templates = {}
chord_names_list = [] # インデックス参照用

# メジャー
for i in range(12):
    name = chroma_labels[i] + 'maj'
    indices = [(i + interval) % 12 for interval in major_intervals]
    all_chord_templates[name] = indices
    chord_names_list.append(name)
# マイナー
for i in range(12):
    name = chroma_labels[i] + 'min'
    indices = [(i + interval) % 12 for interval in minor_intervals]
    all_chord_templates[name] = indices
    chord_names_list.append(name)

vowel_models = train_vowel_models('short.wav')

x_l, _ = librosa.load('Dragon-quest-menu-theme.wav', sr=SR)
x_speech, x_music = separate_speech_music(x_l, SR)

spectrogram = []
estimated_pitch_nn = []
likelihoods_results = [] # 0:a, 1:i, 2:u, 3:e, 4:o
estimated_chord_indices = []

vowel_list = ['a', 'i', 'u', 'e', 'o']
frequencies = np.fft.rfftfreq(SIZE_FRAME, d=1.0 / SR)

for i in np.arange(0, len(x_l) - SIZE_FRAME, SIZE_SHIFT):
    idx = int(i)
    x_frame = x_l[idx:idx + SIZE_FRAME]
    
    # パワー計算
    current_rms = np.sqrt(np.mean(x_frame**2))
    current_db = 20 * np.log10(current_rms + 1e-6)

    # スペクトル計算
    fft_spec = np.fft.rfft(x_frame * HAMMING_WINDOW)
    mag_spec = np.abs(fft_spec)
    fft_log_abs_spec = np.log(mag_spec + 1e-6)
    spectrogram.append(fft_log_abs_spec)

    # ピッチ推定
    best_nn = estimate_pitch_harmonic_sum(fft_log_abs_spec, SR, SIZE_FRAME)
    if current_db < DB_LIM:
        best_nn = -1
    estimated_pitch_nn.append(best_nn)

    # 母音識別
    ceps_feat = calculate_cepstrum_feature(x_frame)
    v_likelihoods = [calculate_log_likelihood(ceps_feat, vowel_models[v][0], vowel_models[v][1]) 
                     for v in vowel_list]
    likelihoods_results.append(np.argmax(v_likelihoods))
    
    # 和音推定
    cv = chroma_vector(mag_spec, frequencies)
    max_likelihood = -float('inf')
    best_chord_idx = -1
    for name, indices in all_chord_templates.items():
        l_h = calculate_likelihood(cv, indices, weights)
        if l_h > max_likelihood:
            max_likelihood = l_h
            best_chord_idx = chord_names_list.index(name)
    
    if max_likelihood > 5.0:
        estimated_chord_indices.append(best_chord_idx)
    else:
        estimated_chord_indices.append(-1)

total_duration = (len(spectrogram) * SIZE_SHIFT) / SR
times = np.linspace(0, total_duration, len(spectrogram))

# 500Hz以下のデータを抽出
idx_500 = np.where(frequencies <= 500)[0]
spec_data = np.array(spectrogram).T[idx_500, :]
f_min, f_max = frequencies[idx_500[0]], frequencies[idx_500[-1]]

# 推定ピッチのHz変換
pitch_hz = [nn2hz(nn) if nn > 0 else np.nan for nn in estimated_pitch_nn]

# グラフと軸の作成
fig, ax = plt.subplots(figsize=(12, 7))
plt.subplots_adjust(bottom=0.25, left=0.2)

# スペクトログラム描画
img = ax.imshow(spec_data, aspect='auto', origin='lower', 
                extent=[times[0], times[-1], f_min, f_max], 
                cmap='magma', vmin=-5, vmax=5, zorder=1)

# ピッチラインとシークバー
pitch_line, = ax.plot(times, pitch_hz, color='cyan', linewidth=2, label='Estimated F0', zorder=2)
v_line = ax.axvline(x=0, color='red', linestyle='--', linewidth=2, zorder=3)
vowel_text = ax.text(0.02, 0.9, '', transform=ax.transAxes, color='yellow', fontsize=18, fontweight='bold', zorder=4)
chord_text = ax.text(0.02, 0.8, '', transform=ax.transAxes, color='orange', fontsize=18, fontweight='bold', zorder=4)

ax.set_ylim(0, 500)
ax.set_title('NMF Separation & Vowel & Chord Analysis')
ax.set_xlabel('Time(s)')
ax.set_ylabel('Frequency (Hz)')
ax.legend(loc='upper right')

class AudioVisualizer:
    def __init__(self, full, speech, music, sr):
        self.audio_map = {"Full": full, "Speech": speech, "Music": music}
        self.current_audio = full
        self.sr = sr
        self.ani = None
        self.is_playing = False
        self.stream = None
        self.current_out_pos = 0

    def set_mode(self, label):
        self.current_audio = self.audio_map[label]

    def _callback(self, outdata, frames, time_info, status):
        if self.is_playing:
            remainder = len(self.current_audio) - self.current_out_pos
            if remainder <= 0:
                outdata.fill(0)
                return
            
            chunk_size = min(frames, remainder)
            audio_chunk = self.current_audio[self.current_out_pos : self.current_out_pos + chunk_size]
            outdata[:chunk_size, 0] = audio_chunk
            
            if chunk_size < frames:
                outdata[chunk_size:, 0] = 0
            
            self.current_out_pos += chunk_size

    def update(self, frame):
        if self.is_playing:
            elapsed_time = self.current_out_pos / self.sr
            if elapsed_time >= total_duration:
                self.stop(None)
                return v_line, vowel_text, chord_text
            
            v_line.set_xdata([elapsed_time])
            frame_idx = int(elapsed_time * SR / SIZE_SHIFT)
            if 0 <= frame_idx < len(likelihoods_results):
                v_idx = likelihoods_results[frame_idx]
                vowel_text.set_text(f"Vowel: {vowel_list[v_idx]}")
                
                c_idx = estimated_chord_indices[frame_idx]
                chord_name = chord_names_list[c_idx] if c_idx >= 0 else "N"
                chord_text.set_text(f"Chord: {chord_name}")
                
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

ax_play = plt.axes([0.7, 0.05, 0.1, 0.075])
btn_play = Button(ax_play, 'Play', color='lightgreen')
ax_stop = plt.axes([0.82, 0.05, 0.1, 0.075])
btn_stop = Button(ax_stop, 'Stop', color='tomato')

btn_play.on_clicked(visualizer.play)
btn_stop.on_clicked(visualizer.stop)

ax_radio = plt.axes([0.02, 0.4, 0.12, 0.2], facecolor='#f0f0f0')
radio = RadioButtons(ax_radio, ('Full', 'Speech', 'Music'))
radio.on_clicked(visualizer.set_mode)

plt.show()