import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
import librosa
import sounddevice as sd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading

SR = 16000
BLOCK_SIZE = 1024  # リアルタイム処理用のバッファサイズ

# --- 音声処理関数群 ---

def generate_sinusoid(sampling_rate, frequency, duration):
    t = np.arange(int(sampling_rate * duration)) / sampling_rate
    return np.sin(2.0 * np.pi * frequency * t)

def tremolo(input_signal, fs, D, R):
    if D <= 0: return input_signal
    t = np.arange(len(input_signal))
    tremolo_envelope = 1.0 + D * np.sin(2.0 * np.pi * R * t / fs)
    return input_signal * tremolo_envelope

def vibrato(input_signal, fs, D, R):
    if D <= 0 or R <= 0: return input_signal
    n = len(input_signal)
    t = np.arange(n)
    delay_samples = D * 100
    tau = delay_samples * np.sin(2.0 * np.pi * R * t / fs)
    indices = t - tau
    indices = np.clip(indices, 0, n - 1)
    return np.interp(indices, t, input_signal)

def calculate_db_profile(signal, sr, size_frame=512, size_shift=160):
    db_list = []
    time_list = []
    for i in np.arange(0, len(signal) - size_frame, size_shift):
        idx = int(i)
        x_frame = signal[idx:idx + size_frame]
        current_rms = np.sqrt(np.mean(x_frame ** 2))
        current_db = 20 * np.log10(current_rms + 1e-12)
        db_list.append(current_db)
        time_list.append(idx / sr)
    return np.array(time_list), np.array(db_list)

# --- メインGUIクラス ---

class AudioGUI:
    def __init__(self, master):
        self.master = master
        master.title('Audio Studio Pro - Live Voice Suite')
        master.geometry('1400x900') 
        self.colors = {
            'bg': '#0a0e27', 'sidebar': '#0f1629', 'card': '#1a1f3a',
            'accent1': '#6366f1', 'success': '#10b981', 'danger': '#ef4444',
            'text': '#e0e7ff', 'text_muted': '#94a3b8', 'border': '#1e293b',
            'warning': '#f59e0b', 'mic': '#06b6d4'
        }
        master.configure(bg=self.colors['bg'])

        # --- 内部状態 ---
        self.sr = SR
        self.orig_signal = None
        self.processed_signal = None
        self.current_signal = None
        self.is_playing = False
        self.is_mic_on = False
        self.play_pos = 0
        self.play_lock = threading.Lock()
        self.update_job = None
        self.window_length = 5.0
        self.stream = None
        
        # リアルタイム用の位相管理
        self.phase_r = 0.0
        self.phase_t = 0.0
        self.phase_v = 0.0

        # --- レイアウト ---
        self.left_sidebar = tk.Frame(master, bg=self.colors['sidebar'], width=350)
        self.left_sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        self.left_sidebar.pack_propagate(False)

        right_main = tk.Frame(master, bg=self.colors['bg'])
        right_main.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # --- サイドバー構成 ---
        tk.Label(self.left_sidebar, text="CONTROL PANEL", fg=self.colors['accent1'],
                 bg=self.colors['sidebar'], font=('Arial', 12, 'bold')).pack(pady=15)
        
        # ファイル読み込み
        self.btn_select = tk.Button(self.left_sidebar, text="📁 Load Audio File", command=self.select_file,
                                   bg=self.colors['accent1'], fg='white', relief='flat', height=2)
        self.btn_select.pack(pady=5, padx=20, fill='x')
        
        # マイク操作ボタン
        mic_frame = tk.Frame(self.left_sidebar, bg=self.colors['sidebar'])
        mic_frame.pack(pady=10, fill='x', padx=20)
        
        self.btn_mic_on = tk.Button(mic_frame, text="🎤 MIC ON", command=self.start_mic,
                                   bg=self.colors['mic'], fg='white', relief='flat', height=2)
        self.btn_mic_on.pack(side=tk.LEFT, expand=True, fill='x', padx=2)
        
        self.btn_mic_stop = tk.Button(mic_frame, text="🛑 STOP MIC", command=self.stop_mic,
                                   bg=self.colors['danger'], fg='white', relief='flat', height=2, state=tk.DISABLED)
        self.btn_mic_stop.pack(side=tk.LEFT, expand=True, fill='x', padx=2)

        # エフェクトスライダー
        self.create_slider("Voice Change Freq (Hz)", 1, 2000, 1, self.update_params_trigger)
        self.create_slider("Tremolo Depth (D)", 0, 1.0, 0.0, self.update_params_trigger, resolution=0.01)
        self.create_slider("Tremolo Rate (R)", 0, 10.0, 0.0, self.update_params_trigger, resolution=0.1)
        self.create_slider("Vibrato Depth (D)", 0, 1.0, 0.0, self.update_params_trigger, resolution=0.01)
        self.create_slider("Vibrato Rate (R)", 0, 10.0, 0.0, self.update_params_trigger, resolution=0.1)

        # 再生コントロール
        control_frame = tk.Frame(self.left_sidebar, bg=self.colors['sidebar'])
        control_frame.pack(pady=20, fill='x', padx=20)
        
        self.btn_play_original = tk.Button(control_frame, text="▶ ORIGINAL", command=self.play_original,
                                 bg=self.colors['warning'], fg='white', state=tk.DISABLED, width=10)
        self.btn_play_original.pack(side=tk.LEFT, expand=True, fill='x', padx=2)
        
        self.btn_play = tk.Button(control_frame, text="▶ PLAY", command=self.play,
                                 bg=self.colors['success'], fg='white', state=tk.DISABLED, width=10)
        self.btn_play.pack(side=tk.LEFT, expand=True, fill='x', padx=2)

        self.btn_stop = tk.Button(control_frame, text="⬛ STOP", command=self.stop,
                                 bg=self.colors['danger'], fg='white', width=10)
        self.btn_stop.pack(side=tk.LEFT, expand=True, fill='x', padx=2)

        # --- グラフ表示部 ---
        self.fig = Figure(figsize=(10, 8), facecolor=self.colors['card'])
        self.fig.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.08, hspace=0.45)
        self.ax_spec = self.fig.add_subplot(3, 1, 1)
        self.ax_db = self.fig.add_subplot(3, 1, 2)
        self.ax_wave = self.fig.add_subplot(3, 1, 3)
        self.axes = [self.ax_spec, self.ax_db, self.ax_wave]
        self.canvas = FigureCanvasTkAgg(self.fig, right_main)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.style_axes()

    def create_slider(self, label, min_val, max_val, default, command, resolution=1.0):
        frame = tk.Frame(self.left_sidebar, bg=self.colors['sidebar'])
        frame.pack(fill='x', padx=20, pady=5)
        tk.Label(frame, text=label, fg=self.colors['text_muted'], bg=self.colors['sidebar'], font=('Arial', 9)).pack(anchor='w')
        slider = tk.Scale(frame, from_=min_val, to=max_val, orient='horizontal', resolution=resolution,
                         bg=self.colors['sidebar'], fg='white', highlightthickness=0,
                         troughcolor=self.colors['card'], command=lambda x: command())
        slider.set(default)
        slider.pack(fill='x')
        name = label.split(" (")[0].lower().replace(" ", "_")
        setattr(self, f"slider_{name}", slider)

    def style_axes(self):
        titles = ['Spectrum Analysis', 'Volume Profile (dB)', 'Waveform Timeline']
        for ax, title in zip(self.axes, titles):
            ax.set_facecolor('#0f1629')
            ax.set_title(title, color='white', loc='left', fontsize=10, fontweight='bold')
            ax.tick_params(colors=self.colors['text_muted'], labelsize=9)
            ax.grid(True, alpha=0.15, color='#475569')

    # --- マイク音声処理 (プロットなし) ---

    def mic_audio_callback(self, indata, outdata, frames, time, status):
        """マイク入力を加工して出力するだけのコールバック"""
        x = indata[:, 0].copy()
        t_array = np.arange(frames) / self.sr
        
        # 1. Voice Change
        freq = self.slider_voice_change_freq.get()
        if freq > 1.0:
            carrier = np.sin(2.0 * np.pi * (self.phase_r + freq * t_array))
            x *= carrier
            self.phase_r = (self.phase_r + freq * frames / self.sr) % 1.0
        
        # 2. Vibrato
        v_d = self.slider_vibrato_depth.get()
        v_r = self.slider_vibrato_rate.get()
        if v_d > 0:
            delay_samples = v_d * 100
            tau = delay_samples * np.sin(2.0 * np.pi * (self.phase_v + v_r * t_array))
            indices = np.arange(frames) - tau
            indices = np.clip(indices, 0, frames - 1)
            x = np.interp(indices, np.arange(frames), x)
            self.phase_v = (self.phase_v + v_r * frames / self.sr) % 1.0
            
        # 3. Tremolo
        t_d = self.slider_tremolo_depth.get()
        t_r = self.slider_tremolo_rate.get()
        if t_d > 0:
            envelope = 1.0 + t_d * np.sin(2.0 * np.pi * (self.phase_t + t_r * t_array))
            x *= envelope
            self.phase_t = (self.phase_t + t_r * frames / self.sr) % 1.0

        outdata[:, 0] = x
        if outdata.shape[1] > 1:
            outdata[:, 1] = x

    def start_mic(self):
        self.stop() # 再生中のものがあれば止める
        self.phase_r = self.phase_t = self.phase_v = 0.0
        try:
            # グラフ更新(update_animation)を呼ばないことで出力のみに専念
            self.stream = sd.Stream(samplerate=self.sr, channels=2, blocksize=BLOCK_SIZE, 
                                   callback=self.mic_audio_callback)
            self.stream.start()
            self.is_mic_on = True
            self.btn_mic_on.config(state=tk.DISABLED)
            self.btn_mic_stop.config(state=tk.NORMAL)
            self.btn_select.config(state=tk.DISABLED)
        except Exception as e:
            print(f"Mic Error: {e}")

    def stop_mic(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_mic_on = False
        self.btn_mic_on.config(state=tk.NORMAL)
        self.btn_mic_stop.config(state=tk.DISABLED)
        self.btn_select.config(state=tk.NORMAL)

    # --- ファイル読み込み・再生処理 (こちらはプロットあり) ---

    def select_file(self):
        path = filedialog.askopenfilename(filetypes=[('WAV', '*.wav')])
        if path:
            threading.Thread(target=self.process_file_thread, args=(path,), daemon=True).start()

    def process_file_thread(self, path):
        y, _ = librosa.load(path, sr=self.sr)
        self.orig_signal = y.astype(np.float32)
        self.apply_effects()
        self.master.after(0, self.finish_loading)

    def finish_loading(self):
        self.draw_initial_plots()
        self.btn_play.config(state=tk.NORMAL)
        self.btn_play_original.config(state=tk.NORMAL)

    def apply_effects(self):
        freq = self.slider_voice_change_freq.get()
        t_d = self.slider_tremolo_depth.get()
        t_r = self.slider_tremolo_rate.get()
        v_d = self.slider_vibrato_depth.get()
        v_r = self.slider_vibrato_rate.get()
        
        if freq > 1:
            sin_wave = generate_sinusoid(self.sr, freq, len(self.orig_signal)/self.sr)
            x_vc = self.orig_signal * sin_wave
        else:
            x_vc = self.orig_signal.copy()
        
        x_trem = tremolo(x_vc, self.sr, t_d, t_r)
        self.processed_signal = vibrato(x_trem, self.sr, v_d, v_r)
        self.times_db, self.dbs = calculate_db_profile(self.processed_signal, self.sr)

    def update_params_trigger(self):
        if self.orig_signal is not None and not self.is_mic_on:
            self.apply_effects()
            self.draw_initial_plots()

    def draw_initial_plots(self):
        self.ax_wave.clear()
        t = np.arange(len(self.processed_signal)) / self.sr
        self.ax_wave.plot(t, self.processed_signal, color='#ec4899', lw=0.5, alpha=0.7)
        self.ax_wave.set_ylim(-1.1, 1.1)
        self.ax_wave.set_xlim(0, self.window_length)
        
        self.ax_db.clear()
        self.ax_db.plot(self.times_db, self.dbs, color='#8b5cf6', lw=1)
        self.ax_db.set_ylim(-60, 5)
        self.ax_db.set_xlim(0, self.window_length)
        
        self.style_axes()
        self.canvas.draw()

    def play_original(self):
        if self.is_playing or self.orig_signal is None: return
        self.current_signal = self.orig_signal
        self._start_playback()

    def play(self):
        if self.is_playing or self.processed_signal is None: return
        self.current_signal = self.processed_signal
        self._start_playback()

    def _start_playback(self):
        self.is_playing = True
        self.btn_play.config(state=tk.DISABLED)
        self.btn_play_original.config(state=tk.DISABLED)
        
        def callback(outdata, frames, time, status):
            with self.play_lock:
                chunk = self.current_signal[self.play_pos : self.play_pos+frames]
                if len(chunk) < frames:
                    outdata[:len(chunk), 0] = chunk
                    outdata[len(chunk):, 0] = 0
                    raise sd.CallbackStop
                else:
                    outdata[:, 0] = chunk
                    self.play_pos += frames

        self.stream = sd.OutputStream(samplerate=self.sr, channels=1, callback=callback, finished_callback=self.stop)
        self.stream.start()
        self.update_animation()

    def update_animation(self):
        if not self.is_playing: return
        with self.play_lock:
            current_sec = self.play_pos / self.sr
        
        x_min = max(0, current_sec - self.window_length * 0.2)
        x_max = x_min + self.window_length
        self.ax_wave.set_xlim(x_min, x_max)
        self.ax_db.set_xlim(x_min, x_max)
        
        n_fft = 1024
        start = max(0, self.play_pos - n_fft)
        frame = self.current_signal[start : start + n_fft]
        if len(frame) == n_fft:
            spec = np.abs(np.fft.rfft(frame * np.hanning(n_fft)))
            log_spec = 20 * np.log10(spec + 1e-9)
            freqs = np.fft.rfftfreq(n_fft, 1/self.sr)
            self.ax_spec.clear()
            self.ax_spec.fill_between(freqs, log_spec, -100, color='#6366f1', alpha=0.5)
            self.ax_spec.set_ylim(-80, 20)
            self.style_axes()

        self.canvas.draw_idle()
        self.update_job = self.master.after(50, self.update_animation)

    def stop(self):
        if self.is_mic_on: self.stop_mic()
        self.is_playing = False
        self.play_pos = 0
        if self.stream: 
            try: self.stream.stop()
            except: pass
        self.master.after(0, lambda: self.btn_play.config(state=tk.NORMAL))
        self.master.after(0, lambda: self.btn_play_original.config(state=tk.NORMAL))
        if self.update_job: self.master.after_cancel(self.update_job)

if __name__ == '__main__':
    root = tk.Tk()
    app = AudioGUI(root)
    root.mainloop()