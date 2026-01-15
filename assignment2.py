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
import time 

# 定数設定
SR = 16000  # サンプリング周波数
BLOCK_SIZE = 1024  # マイク入力時のバッファサイズ
# アプリケーション全体の配色テーマ
COLOR_THEME = {
    'bg': '#0a0e27', 'sidebar': '#0f1629', 'card': '#1a1f3a',
    'accent1': '#6366f1', 'success': '#10b981', 'danger': '#ef4444',
    'text': '#e0e7ff', 'text_muted': '#94a3b8', 'border': '#1e293b',
    'warning': '#f59e0b', 'mic': '#06b6d4'
}


# 正弦波生成関数(ボイスチェンジ用)
def generate_sinusoid(sampling_rate, frequency, duration):
    t = np.arange(int(sampling_rate * duration)) / sampling_rate
    return np.sin(2.0 * np.pi * frequency * t)

# トレモロ
def tremolo(input_signal, fs, D, R):
    if D <= 0: return input_signal  # 深度が0なら何もしない
    t = np.arange(len(input_signal)) 
    tremolo_envelope = 1.0 + D * np.sin(2.0 * np.pi * R * t / fs)
    return input_signal * tremolo_envelope  

# ビブラート
def vibrato(input_signal, fs, D, R):
    if D <= 0 or R <= 0: return input_signal  # パラメータが無効なら何もしない
    n = len(input_signal)
    t = np.arange(n)  
    delay_samples = D * 100  # 遅延の深さをサンプル数に変換(係数100は調整値)
    tau = delay_samples * np.sin(2.0 * np.pi * R * t / fs)
    indices = t - tau  
    indices = np.clip(indices, 0, n - 1)  # 配列の範囲外に出ないように制限
    # 線形補間を使って、整数でないインデックスの値を推定して波形を再構築
    return np.interp(indices, t, input_signal)

# dB
def calculate_db_profile(signal, sr, size_frame=512, size_shift=160):
    db_list = []  
    time_list = [] 
    # 音声データをフレームに区切って処理
    for i in np.arange(0, len(signal) - size_frame, size_shift):
        idx = int(i)
        x_frame = signal[idx:idx + size_frame]  # フレームの切り出し
        current_rms = np.sqrt(np.mean(x_frame ** 2))
        # e-12はlog(0)のエラー回避用の微小値
        current_db = 20 * np.log10(current_rms + 1e-12)
        db_list.append(current_db)
        time_list.append(idx / sr)  # 現在の時間を記録
    return np.array(time_list), np.array(db_list)


# ファイル読み込み中に表示するローディング画面
class LoadingOverlay(tk.Toplevel):
    def __init__(self, parent, message="Processing Audio File..."):
        super().__init__(parent)
        self.configure(bg=COLOR_THEME['card'])  # 背景色設定
        self.overrideredirect(True)  # タイトルバーを消してスタイリッシュにする
        
        # 画面中央に配置するための計算
        width, height = 400, 150
        p_w = parent.winfo_width()
        p_h = parent.winfo_height()
        p_x = parent.winfo_x()
        p_y = parent.winfo_y()
        self.geometry(f"{width}x{height}+{p_x + p_w//2 - width//2}+{p_y + p_h//2 - height//2}")

        # メッセージラベル
        tk.Label(self, text=message, font=("Arial", 12, "bold"), 
                 bg=COLOR_THEME['card'], fg=COLOR_THEME['text']).pack(pady=(30, 10))
        
        # プログレスバー(行ったり来たりするアニメーション)
        self.progress = ttk.Progressbar(self, orient=tk.HORIZONTAL, length=300, mode='indeterminate')
        self.progress.pack(pady=10)
        self.progress.start(10)  # アニメーション開始
        
        # 枠線の装飾と最前面表示設定
        self.config(highlightbackground=COLOR_THEME['accent1'], highlightthickness=2)
        self.attributes("-topmost", True)
        self.grab_set()  # 親ウィンドウの操作をロックする

# アプリ起動時のロゴ表示画面
class SplashScreen(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.configure(bg=COLOR_THEME['bg'])
        self.overrideredirect(True)  # タイトルバーなし
        # 画面サイズと配置計算
        width, height = 600, 350
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

        # 外枠と内枠を作ってデザイン性を高める
        outer_frame = tk.Frame(self, bg=COLOR_THEME['accent1'], padx=2, pady=2)
        outer_frame.pack(fill=tk.BOTH, expand=True)
        inner_frame = tk.Frame(outer_frame, bg=COLOR_THEME['bg'])
        inner_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # アプリ名などの表示
        tk.Label(inner_frame, text="AUDIO STUDIO PRO", font=("Helvetica", 32, "bold"), 
                 bg=COLOR_THEME['bg'], fg=COLOR_THEME['text']).pack(pady=(60, 10))
        tk.Label(inner_frame, text="Live Voice", font=("Helvetica", 16), 
                 bg=COLOR_THEME['bg'], fg=COLOR_THEME['accent1']).pack(pady=(0, 50))
        self.status_label = tk.Label(inner_frame, text="Initializing...", bg=COLOR_THEME['bg'], fg=COLOR_THEME['text_muted'])
        self.status_label.pack()

        # 読み込みバー
        self.progress = ttk.Progressbar(inner_frame, length=400, mode='indeterminate')
        self.progress.pack(pady=20)
        self.progress.start(15)
        self.update()  # 画面描画を更新

# メインのGUIクラス
class AudioGUI:
    # 初期化処理
    def __init__(self, master):
        self.master = master
        master.title('Audio Studio Pro')  # ウィンドウタイトル
        master.geometry('1400x900')   # ウィンドウサイズ
        self.colors = COLOR_THEME
        master.configure(bg=self.colors['bg'])  # 背景色適用

        # 内部変数の初期化
        self.sr = SR
        self.orig_signal = None  # 元の音声データ
        self.processed_signal = None  # エフェクト適用後の音声データ
        self.current_signal = None  # 現在再生対象のデータ
        self.is_playing = False     # 再生中フラグ
        self.is_mic_on = False      # マイク使用中フラグ
        self.play_pos = 0           # 現在の再生位置(サンプル単位)
        self.play_lock = threading.Lock()  # スレッド競合を防ぐためのロック
        self.update_job = None      # アニメーション更新用タイマーID
        self.window_length = 5.0    # グラフの表示幅(秒)
        self.stream = None          # サウンドデバイスのストリームオブジェクト
        self.loading_overlay = None
        
        # マイク入力時の位相管理用変数(音が切れないようにするため)
        self.phase_r = 0.0  # リング変調用位相
        self.phase_t = 0.0  # トレモロ用位相
        self.phase_v = 0.0  # ビブラート用位相

        self.setup_ui()  # UI構築メソッドの呼び出し

    # UIの配置と構築
    def setup_ui(self):
        # 左側のサイドバー(コントロールパネル)
        self.left_sidebar = tk.Frame(self.master, bg=self.colors['sidebar'], width=350)
        self.left_sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        self.left_sidebar.pack_propagate(False)  # サイズを固定

        # 右側のメインエリア(グラフ表示用)
        right_main = tk.Frame(self.master, bg=self.colors['bg'])
        right_main.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # サイドバーのタイトル
        tk.Label(self.left_sidebar, text="CONTROL PANEL", fg=self.colors['accent1'],
                 bg=self.colors['sidebar'], font=('Arial', 12, 'bold')).pack(pady=15)
        
        # ファイル読み込みボタン
        self.btn_select = tk.Button(self.left_sidebar, text="📁 Load Audio File", command=self.select_file,
                                   bg=self.colors['accent1'], fg='white', relief='flat', height=2)
        self.btn_select.pack(pady=5, padx=20, fill='x')
        
        # マイク操作ボタン群のフレーム
        mic_frame = tk.Frame(self.left_sidebar, bg=self.colors['sidebar'])
        mic_frame.pack(pady=10, fill='x', padx=20)
        # マイクONボタン
        self.btn_mic_on = tk.Button(mic_frame, text="🎤 MIC ON", command=self.start_mic, bg=self.colors['mic'], fg='white', relief='flat', height=2)
        self.btn_mic_on.pack(side=tk.LEFT, expand=True, fill='x', padx=2)
        # マイクOFFボタン
        self.btn_mic_stop = tk.Button(mic_frame, text="🛑 STOP MIC", command=self.stop_mic, bg=self.colors['danger'], fg='white', relief='flat', height=2, state=tk.DISABLED)
        self.btn_mic_stop.pack(side=tk.LEFT, expand=True, fill='x', padx=2)

        # パラメータ調整用スライダーの作成
        # ボイスチェンジ周波数、トレモロ深度・速度、ビブラート深度・速度
        self.create_slider("Voice Change Freq (Hz)", 1, 2000, 1, self.update_params_trigger)
        self.create_slider("Tremolo Depth (D)", 0, 1.0, 0.0, self.update_params_trigger, resolution=0.01)
        self.create_slider("Tremolo Rate (R)", 0, 10.0, 0.0, self.update_params_trigger, resolution=0.1)
        self.create_slider("Vibrato Depth (D)", 0, 1.0, 0.0, self.update_params_trigger, resolution=0.01)
        self.create_slider("Vibrato Rate (R)", 0, 10.0, 0.0, self.update_params_trigger, resolution=0.1)

        # 再生コントロールボタン群の配置
        control_frame = tk.Frame(self.left_sidebar, bg=self.colors['sidebar'])
        control_frame.pack(pady=20, fill='x', padx=20)
        self.btn_play_original = tk.Button(control_frame, text="▶ ORIGINAL", command=self.play_original, bg=self.colors['warning'], fg='white', state=tk.DISABLED, width=10)
        self.btn_play_original.pack(side=tk.LEFT, expand=True, fill='x', padx=2)
        self.btn_play = tk.Button(control_frame, text="▶ PLAY", command=self.play, bg=self.colors['success'], fg='white', state=tk.DISABLED, width=10)
        self.btn_play.pack(side=tk.LEFT, expand=True, fill='x', padx=2)
        self.btn_stop = tk.Button(control_frame, text="⬛ STOP", command=self.stop, bg=self.colors['danger'], fg='white', width=10)
        self.btn_stop.pack(side=tk.LEFT, expand=True, fill='x', padx=2)

        # Matplotlibのグラフ領域設定
        # 3行1列のグラフを作成
        self.fig = Figure(figsize=(10, 8), facecolor=self.colors['card'])
        self.fig.subplots_adjust(left=0.1, right=0.95, top=0.93, bottom=0.1, hspace=0.6) # 余白調整
        self.ax_spec = self.fig.add_subplot(3, 1, 1)  # 上段：スペクトル
        self.ax_db = self.fig.add_subplot(3, 1, 2)    # 中段：dB
        self.ax_wave = self.fig.add_subplot(3, 1, 3)  # 下段：波形
        self.axes = [self.ax_spec, self.ax_db, self.ax_wave]
        
        # FigureをTkinterキャンバスに埋め込む
        self.canvas = FigureCanvasTkAgg(self.fig, right_main)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.style_axes() # グラフの色やラベルの初期設定

    # スライダー作成用のヘルパー関数
    def create_slider(self, label, min_val, max_val, default, command, resolution=1.0):
        frame = tk.Frame(self.left_sidebar, bg=self.colors['sidebar'])
        frame.pack(fill='x', padx=20, pady=5)
        tk.Label(frame, text=label, fg=self.colors['text_muted'], bg=self.colors['sidebar'], font=('Arial', 9)).pack(anchor='w')
        # スライダー本体
        slider = tk.Scale(frame, from_=min_val, to=max_val, orient='horizontal', resolution=resolution,
                          bg=self.colors['sidebar'], fg='white', highlightthickness=0,
                          troughcolor=self.colors['card'], command=lambda x: command()) # 値変更時にcommandを実行
        slider.set(default)
        slider.pack(fill='x')
        # インスタンス変数として保存(後で値を取得するため)
        name = label.split(" (")[0].lower().replace(" ", "_")
        setattr(self, f"slider_{name}", slider)

    # グラフの軸ラベルや色の設定
    def style_axes(self):
        axis_info = [
            ('Spectrum Analysis', 'Frequency [Hz]', 'Magnitude [dB]'),
            ('Volume Profile (dB)', 'Time [s]', 'Level [dB]'),
            ('Waveform Timeline', 'Time [s]', 'Amplitude')
        ]
        for ax, (title, xl, yl) in zip(self.axes, axis_info):
            ax.set_facecolor('#0f1629') # プロットエリアの背景色
            ax.set_title(title, color='white', loc='left', fontsize=10, fontweight='bold')
            ax.set_xlabel(xl, color=self.colors['text_muted'], fontsize=8)
            ax.set_ylabel(yl, color=self.colors['text_muted'], fontsize=8)
            ax.tick_params(colors=self.colors['text_muted'], labelsize=9) # 目盛りの色
            ax.grid(True, alpha=0.15, color='#475569') # グリッド線

    # ファイル選択時の処理
    def select_file(self):
        path = filedialog.askopenfilename(filetypes=[('WAV', '*.wav')])
        if path:
            self.loading_overlay = LoadingOverlay(self.master, "Analyzing Waveform...")
            # UIを止めないように別スレッドで読み込み実行
            threading.Thread(target=self.process_file_thread, args=(path,), daemon=True).start()

    # 別スレッドでのファイル読み込みとエフェクト適用
    def process_file_thread(self, path):
        try:
            y, _ = librosa.load(path, sr=self.sr) # Librosaで読み込み
            self.orig_signal = y.astype(np.float32) # float32に変換
            self.apply_effects() # 現在のスライダー値に基づいてエフェクト適用
        except Exception as e:
            print(f"File Load Error: {e}")
        finally:
            # 処理完了をメインスレッドに通知
            self.master.after(0, self.finish_loading)

    # 読み込み完了後のGUI更新
    def finish_loading(self):
        if self.loading_overlay:
            self.loading_overlay.destroy() # ローディング画面を消す
            self.loading_overlay = None
        
        self.draw_initial_plots() # グラフを描画
        self.btn_play.config(state=tk.NORMAL) # 再生ボタンを有効化
        self.btn_play_original.config(state=tk.NORMAL)

    # 現在のスライダー値を取得してエフェクトを適用する
    def apply_effects(self):
        if self.orig_signal is None: return
        # スライダーの値を取得
        freq = self.slider_voice_change_freq.get()
        t_d = self.slider_tremolo_depth.get()
        t_r = self.slider_tremolo_rate.get()
        v_d = self.slider_vibrato_depth.get()
        v_r = self.slider_vibrato_rate.get()
        
        # ボイスチェンジ
        if freq > 1:
            sin_wave = generate_sinusoid(self.sr, freq, len(self.orig_signal)/self.sr)
            x_vc = self.orig_signal * sin_wave # 信号同士の掛け算
        else:
            x_vc = self.orig_signal.copy()
        
        # トレモロ,ビブラートの順に適用
        x_trem = tremolo(x_vc, self.sr, t_d, t_r)
        self.processed_signal = vibrato(x_trem, self.sr, v_d, v_r)
        # dBを事前計算しておく
        self.times_db, self.dbs = calculate_db_profile(self.processed_signal, self.sr)

    # 静的なグラフ(dBと波形)の初期描画
    def draw_initial_plots(self):
        self.ax_wave.clear()
        t = np.arange(len(self.processed_signal)) / self.sr
        # 波形のプロット
        self.ax_wave.plot(t, self.processed_signal, color='#ec4899', lw=0.5, alpha=0.7)
        self.ax_wave.set_ylim(-1.1, 1.1)
        self.ax_wave.set_xlim(0, self.window_length)
        
        self.ax_db.clear()
        # dBのプロット
        self.ax_db.plot(self.times_db, self.dbs, color='#8b5cf6', lw=1)
        self.ax_db.set_ylim(-60, 5)
        self.ax_db.set_xlim(0, self.window_length)
        
        self.style_axes()
        self.canvas.draw()

    # スライダー操作時のコールバック(マイクオフ時のみファイルを再計算)
    def update_params_trigger(self):
        if self.orig_signal is not None and not self.is_mic_on:
            self.apply_effects()
            self.draw_initial_plots()

    # マイク入力開始処理
    def start_mic(self):
        self.stop() # 再生中のものがあれば停止
        self.phase_r = self.phase_t = self.phase_v = 0.0 # 位相リセット
        try:
            # 入力ストリームを開く。callbackに関数を指定することで、データが入るたびに呼び出される。
            self.stream = sd.Stream(samplerate=self.sr, channels=2, blocksize=BLOCK_SIZE, callback=self.mic_audio_callback)
            self.stream.start()
            self.is_mic_on = True
            self.btn_mic_on.config(state=tk.DISABLED)
            self.btn_mic_stop.config(state=tk.NORMAL)
            self.btn_select.config(state=tk.DISABLED) # ファイル読み込み無効化
        except: pass

    # マイク入力時のリアルタイム処理コールバック
    def mic_audio_callback(self, indata, outdata, frames, time, status):
        x = indata[:, 0].copy() # 入力データの取得
        t_array = np.arange(frames) / self.sr
        freq = self.slider_voice_change_freq.get()
        # ボイスチェンジ処理
        if freq > 1.0:
            # 連続したサイン波を作るために、前回の位相(phase_r)を引き継ぐ
            carrier = np.sin(2.0 * np.pi * (self.phase_r + freq * t_array))
            x *= carrier
            self.phase_r = (self.phase_r + freq * frames / self.sr) % 1.0 # 位相更新
            
        t_depth = self.slider_tremolo_depth.get()
        t_rate = self.slider_tremolo_rate.get()
        if t_depth > 0:
            # 位相(phase_t)を使って連続性を保つ
            trem_env = 1.0 + t_depth * np.sin(2.0 * np.pi * (self.phase_t + t_rate * t_array))
            x *= trem_env
            self.phase_t = (self.phase_t + t_rate * frames / self.sr) % 1.0
            
        # 出力バッファに書き込む＝スピーカーから音が出る
        outdata[:, 0] = x
        if outdata.shape[1] > 1: outdata[:, 1] = x

    # マイク停止
    def stop_mic(self):
        if self.stream: self.stream.stop(); self.stream.close(); self.stream = None
        self.is_mic_on = False
        self.btn_mic_on.config(state=tk.NORMAL)
        self.btn_mic_stop.config(state=tk.DISABLED)
        self.btn_select.config(state=tk.NORMAL)

    # オリジナル音声再生
    def play_original(self):
        if self.is_playing or self.orig_signal is None: return
        self.current_signal = self.orig_signal
        self._start_playback()

    # 加工後音声再生
    def play(self):
        if self.is_playing or self.processed_signal is None: return
        self.current_signal = self.processed_signal
        self._start_playback()

    # 再生開始共通メソッド
    def _start_playback(self):
        self.is_playing = True
        self.btn_play.config(state=tk.DISABLED)
        self.btn_play_original.config(state=tk.DISABLED)
        
        # 再生用コールバック
        def callback(outdata, frames, time, status):
            with self.play_lock: # 変数競合を防ぐ
                # 現在位置から必要なフレーム数分切り出し
                chunk = self.current_signal[self.play_pos : self.play_pos+frames]
                if len(chunk) < frames: # データが足りない＝再生終了
                    outdata[:len(chunk), 0] = chunk; outdata[len(chunk):, 0] = 0
                    raise sd.CallbackStop # ストリーム停止指令
                else:
                    outdata[:, 0] = chunk; self.play_pos += frames # 再生位置を進める
        
        self.stream = sd.OutputStream(samplerate=self.sr, channels=1, callback=callback, finished_callback=self.stop)
        self.stream.start()
        self.update_animation() # アニメーションループ開始

    # 再生中のアニメーション更新
    def update_animation(self):
        if not self.is_playing: return
        with self.play_lock: current_sec = self.play_pos / self.sr
        
        # グラフの表示範囲を現在の再生位置に合わせてスクロールさせる
        x_min = max(0, current_sec - self.window_length * 0.2)
        x_max = x_min + self.window_length
        self.ax_wave.set_xlim(x_min, x_max); self.ax_db.set_xlim(x_min, x_max)
        
        # 以前の赤い縦線(再生バー)を消去
        try:
            if self.playback_line_wave: self.playback_line_wave.remove()
            if self.playback_line_db: self.playback_line_db.remove()
        except: pass
        
        # 新しい赤い縦線を描画
        self.playback_line_wave = self.ax_wave.axvline(x=current_sec, color='#ef4444', linewidth=2, linestyle='-', alpha=0.8)
        self.playback_line_db = self.ax_db.axvline(x=current_sec, color='#ef4444', linewidth=2, linestyle='-', alpha=0.8)
        
        # スペクトル分析(FFT)の更新
        n_fft = 1024
        start = max(0, self.play_pos - n_fft)
        frame = self.current_signal[start : start + n_fft]
        if len(frame) == n_fft:
            # ハニング窓を掛けてFFT実行
            spec = np.abs(np.fft.rfft(frame * np.hanning(n_fft)))
            # 対数変換してdBにする
            log_spec = 20 * np.log10(spec + 1e-9)
            freqs = np.fft.rfftfreq(n_fft, 1/self.sr) # 周波数軸データ
            
            self.ax_spec.clear()
            # スペクトルを塗りつぶしグラフで描画
            self.ax_spec.fill_between(freqs, log_spec, -100, color='#6366f1', alpha=0.5)
            self.ax_spec.set_ylim(-80, 20)
            self.style_axes() # スタイル再適用
            
        self.canvas.draw_idle() # 描画更新
        # 50ミリ秒後に自分自身を再度呼び出す
        self.update_job = self.master.after(50, self.update_animation)
    
    # 停止処理
    def stop(self):
        if self.is_mic_on: self.stop_mic()
        self.is_playing = False; self.play_pos = 0
        if self.stream: 
            try: self.stream.stop()
            except: pass
        # ボタン状態のリセット
        self.master.after(0, lambda: self.btn_play.config(state=tk.NORMAL))
        self.master.after(0, lambda: self.btn_play_original.config(state=tk.NORMAL))
        if self.update_job: self.master.after_cancel(self.update_job)

if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw() 
    splash = SplashScreen(root)
    time.sleep(1.0) 
    app = AudioGUI(root) 
    splash.destroy() 
    root.deiconify() 
    root.mainloop() 