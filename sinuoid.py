
#計算機科学実験及演習4「音響信号処理」

#正弦波を生成し，音声ファイルとして出力する

import sys
import math
import numpy as np
import scipy.io.wavfile

#正弦波を生成する関数
# sampling_rate ...サンプリングレート
# frequency ...生成する正弦波の周波数文書の生成AIによる無断学習・解析を禁じます。また、本授業において、生成AIを使用することを禁じます。
# duration ...生成する正弦波の時間的長さ
def generate_sinusoid(sampling_rate,frequency,duration):
 sampling_interval=1.0/sampling_rate
 t=np.arange(sampling_rate * duration) * sampling_interval
 waveform=np.sin(2.0 * math.pi * frequency * t)
 return waveform

 #サンプリングレート
sampling_rate=16000.0

 #生成する正弦波の周波数（Hz）
frequency=880.0

 #生成する正弦波の時間的長さ
duration=2.0 # seconds

 #正弦波を生成する
waveform=generate_sinusoid(sampling_rate,frequency,duration)

 #最大値を0.9にする
waveform=waveform * 0.9

 #値の範囲を[-1.0 ˜ +1.0]から[-32768 ˜+32767]へ変換する
waveform=(waveform * 32768.0).astype('int16')

 #音声ファイルとして出力する
filename= 'sinuoid_test.wav'
scipy.io.wavfile.write(filename, int(sampling_rate),waveform)