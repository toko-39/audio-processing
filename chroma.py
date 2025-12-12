import math
import matplotlib.pyplot as plt
import numpy as np
import librosa

SR= 16000
# x,_ =librosa.load('short.wav',sr=SR)
x_l,_ =librosa.load('free_10sec.wav',sr=SR)

size_frame= 2048 #2のべき乗

hamming_window = np.hamming(size_frame)

 #シフトサイズ
size_shift= 16000/100 #0.01秒(10 msec)

def nn2hz(notenum):
    return 440.0 * (2.0 ** ((notenum-69) / 12.0))

 #周波数からノートナンバーへ
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

spectrogram=[]
chromagram = []

# frequencies =np.linspace(8000/size_shift, 8000, size_shift)
frequencies = np.fft.rfftfreq(size_frame, d=1.0 / SR)

for i in np.arange(0, len(x_l)-size_frame,size_shift):

    idx= int(i)
    x_frame=x_l[idx:idx+size_frame]

    fft_spec = np.fft.rfft(x_frame * hamming_window)

    fft_log_abs_spec=np.log(np.abs(fft_spec) + 1e-6) 

    spectrogram.append(fft_log_abs_spec)
    
    cv = np.zeros(12)
    chromagram.append(chroma_vector(fft_log_abs_spec, frequencies))

a_root, a_3rd, a_5th = 1.0, 0.5, 0.8
weights = [a_root, a_3rd, a_5th]

major_intervals = [0, 4, 7] 
minor_intervals = [0, 3, 7]

chroma_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
all_chord_templates = {} # {コード名: 構成音インデックス}
chord_to_index = {} # {コード名: インデックス (0-23)}

index_counter = 0
for root_index in range(12):
    # メジャーコード
    major_name = chroma_labels[root_index] + 'maj'
    major_indices = [(root_index + interval) % 12 for interval in major_intervals]
    all_chord_templates[major_name] = major_indices
    chord_to_index[major_name] = index_counter
    index_counter += 1

for root_index in range(12):
    #  マイナーコード
    minor_name = chroma_labels[root_index] + 'min'
    minor_indices = [(root_index + interval) % 12 for interval in minor_intervals]
    all_chord_templates[minor_name] = minor_indices
    chord_to_index[minor_name] = index_counter
    index_counter += 1

def calculate_likelihood(cv, indices, weights):
    likelihood = 0.0
    for weight, index in zip(weights, indices):
        # クロマベクトルcvの要素を重み付けして合計
        likelihood += weight * cv[index]
    return likelihood

estimated_chord_indices = [] # 推定された和音インデックス (0-23, または-1)

# クロマグラムの各フレームについて和音を推定
for frame_cv in chromagram:
    max_likelihood = -float('inf')
    best_index = -1 # 初期値は未検出
    
    # 24個の全てのコードテンプレートと比較
    for name, indices in all_chord_templates.items():
        likelihood = calculate_likelihood(frame_cv, indices, weights)
        
        if likelihood > max_likelihood:
            max_likelihood = likelihood
            best_index = chord_to_index[name]
            
    # ノイズや無音を無視するための閾値設定
    if max_likelihood > 5.0: 
        estimated_chord_indices.append(best_index)
    else:
        # 和音なし (N) の場合は -1 を使用
        estimated_chord_indices.append(-1) 


chromagram_array = np.array(chromagram).T
spectrogram_array = np.array(spectrogram).T
duration = len(x_l) / SR # 信号の総時間 (秒)

fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
time_points = np.linspace(0, duration, len(estimated_chord_indices))

ax1 = axes[0]
ax1.set_title('Spectrogram')
ax1.set_ylabel('Frequency [Hz]')
im1 = ax1.imshow(
    np.flipud(spectrogram_array),
    extent=[0, duration, 0, SR / 2],
    aspect='auto',
    interpolation='nearest',
    cmap='viridis'
)
ax1.set_ylim(0, 300) 


ax2 = axes[1]
ax2.set_title('Chromagram')
ax2.set_ylabel('Chroma')
im2 = ax2.imshow(
    chromagram_array,
    extent=[0, duration, 0, 12],
    aspect='auto',
    interpolation='nearest',
    cmap='plasma',
    origin='lower'
)
ax2.set_yticks(np.arange(0.5, 12.5, 1))
ax2.set_yticklabels(chroma_labels)
ax2.set_ylim(0, 12)


ax3 = axes[2]
ax3.set_title('Estimated Chord Index')
ax3.set_xlabel('Time [sec]')
ax3.set_ylabel('Chord Index (0-23)')

# グラフの描画
ax3.plot(time_points, estimated_chord_indices, drawstyle='steps-post', color='red', linewidth=1.5)

# 縦軸ラベルの設定
# 0-23のインデックスに対応するコード名リストを作成
index_to_chord_name = {v: k for k, v in chord_to_index.items()}
y_ticks = list(range(24))

# 和音なし(-1)を見やすくするため、縦軸の下限を-1にする
ax3.set_ylim(-1.5, 23.5)
ax3.set_yticks(y_ticks)
ax3.set_yticklabels([index_to_chord_name[i] for i in y_ticks], fontsize=8)


# プロットの表示
plt.tight_layout()
plt.show()