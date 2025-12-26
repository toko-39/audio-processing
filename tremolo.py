import numpy as np
import matplotlib.pyplot as plt
import librosa
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
x, _ = librosa.load(selected_path, sr=SR)
D = 0.5
R = 8.0

def tremolo(input_signal, fs, D, R):
    t = np.arange(len(input_signal))
    tremolo_envelope = 1.0 + D * np.sin(2.0 * np.pi * R * t / fs)
    return input_signal * tremolo_envelope

x_tremolo = tremolo(x, SR, D, R)

def calculate_db_profile(signal, sr, size_frame=512, size_shift=160):
    db_list = []
    time_list = []
    
    for i in np.arange(0, len(signal) - size_frame, size_shift):
        idx = int(i)
        x_frame = signal[idx:idx + size_frame]
        
        current_rms = np.sqrt(np.mean(x_frame**2))
        current_db = 20 * np.log10(current_rms + 1e-12)
        
        db_list.append(current_db)
        time_list.append(idx / sr)
        
    return time_list, db_list

size_frame = 512
size_shift = 160 

t_orig, db_orig = calculate_db_profile(x, SR, size_frame, size_shift)
t_trem, db_trem = calculate_db_profile(x_tremolo, SR, size_frame, size_shift)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
fig.subplots_adjust(hspace=0.4)

full_time_axis = np.linspace(0, len(x) / SR, len(x))

ax1.plot(full_time_axis, x, color='silver', label='Original', alpha=0.6)
ax1.plot(full_time_axis, x_tremolo, color='crimson', label='Tremolo', alpha=0.7)
ax1.set_title(f'Full Waveform Comparison (File: {selected_path.split("/")[-1]})')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Amplitude')
ax1.set_xlim(0, full_time_axis[-1])
ax1.legend(loc='upper right')

ax2.plot(t_orig, db_orig, color='gray', label='Original dB', alpha=0.5)
ax2.plot(t_trem, db_trem, color='green', label='Tremolo dB')
ax2.set_title(f'Volume Level(D={D}, R={R}Hz)')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Level [dB]')
ax2.set_xlim(0, full_time_axis[-1])
ax2.set_ylim(np.max(db_trem)-40, np.max(db_trem)+5)
ax2.legend(loc='upper right')

plt.show()