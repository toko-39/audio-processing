import librosa
import numpy as np
from scipy.io import wavfile
import os

# --- 設定 ---
input_wav = "free.wav" 
output_wav = "free_10sec.wav" 
SR = 16000 # サンプリングレート (librosaのデフォルトは22050だが、ここでは16000を使用)
start_time_sec = 0  # 開始時刻（秒）最初から
duration_sec = 10   # 切り出す長さ（秒）
# ----------------

def trim_wav_with_librosa(input_path, output_path, sr, start_sec, duration_sec):
    try:
        # librosaで音声データを読み込む（NumPy配列として）
        y, actual_sr = librosa.load(input_path, sr=sr)
        
        # サンプル数に変換
        start_sample = int(start_sec * actual_sr)
        end_sample = int((start_sec + duration_sec) * actual_sr)
        
        # ファイルの長さを超えないように調整
        if end_sample > len(y):
            end_sample = len(y)
            print(f"⚠️ 注意: ファイルが短いため、末尾まで ({len(y)/actual_sr:.2f}秒) 切り出します。")
            
        # 切り出し (NumPyのスライス操作)
        trimmed_y = y[start_sample:end_sample]
        
        # scipy.io.wavfileを使ってWAVファイルとして保存
        # wavfileは通常16bit整数形式を扱うため、データを正規化して16bitに変換
        
        # データを-32768から32767の範囲にスケーリング
        trimmed_y_int16 = (trimmed_y * 32767).astype(np.int16)
        
        wavfile.write(output_path, actual_sr, trimmed_y_int16)
        
        print(f"✅ 成功: {start_sec}秒目から{duration_sec}秒間を切り出し、{output_path}に保存しました。")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")

# 実行
trim_wav_with_librosa(input_wav, output_wav, SR, start_time_sec, duration_sec)