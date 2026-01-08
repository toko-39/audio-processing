import numpy as np
import matplotlib.pyplot as plt
import librosa
import scipy.io.wavfile
import tkinter as tk
from tkinter import filedialog
import sys
from scipy.interpolate import interp1d

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
x, _ = librosa.load(selected_path, sr=SR)

# --- パラメータ設定 ---
D = 15.0   # 遅延時間の深さ（時間を揺らす大きさ）
R = 6.0    # 揺れの頻度 (Hz)

# --- ビブラートの実装 ---
def vibrato(input_signal, fs, D, R):
    n_samples = len(input_signal)
    t = np.arange(n_samples)
    
    # 遅延時間 tau(t) の計算
    tau = D * np.sin(2.0 * np.pi * R * t / fs)
    
    # 参照インデックスの計算 (t - tau)
    indices = t - tau
    indices = np.clip(indices, 0, n_samples - 1)
    
    # 線形補間により、整数ではない位置の値を推定
    f_interp = interp1d(t, input_signal, kind='linear', fill_value="extrapolate")
    return f_interp(indices)

# 処理実行
x_vibrato = vibrato(x, SR, D, R)

# --- 音声の保存 ---
# 出力用にスケーリングして int16 に変換
x_vibrato_int = (x_vibrato * 32767.0).astype(np.int16)
output_filename = "vibrato_output.wav"
scipy.io.wavfile.write(output_filename, SR, x_vibrato_int)
print(f"加工後の音声を保存しました: {output_filename}")

# --- プロット部分（2段構成） ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
fig.subplots_adjust(hspace=0.4)

full_time = np.linspace(0, len(x) / SR, len(x))

# 1. 全体波形：重なりを見やすく
ax1.plot(full_time, x, color='silver', label='Original', alpha=0.5, lw=0.8)
ax1.plot(full_time, x_vibrato, color='royalblue', label='Vibrato', alpha=0.7, lw=0.8)
ax1.set_title(f'Vibrato Waveform Comparison (D={D}, R={R}Hz)')
ax1.set_ylabel('Amplitude')
ax1.set_xlim(0, full_time[-1])
ax1.legend(loc='upper right')

# 2. 差分波形：変化した成分のみを抽出
# 元の音と加工後の音の差を表示することで、どの程度時間がズレたかを可視化
diff = x - x_vibrato
ax2.plot(full_time, diff, color='crimson', lw=0.5)
ax2.set_title('Difference (Original - Vibrato) - Highlights the effect of time-shifting')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Diff Amplitude')
ax2.set_xlim(0, full_time[-1])

plt.show()