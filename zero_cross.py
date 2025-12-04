import matplotlib.pyplot as plt
import numpy as np
import librosa

SR= 16000

x,_ =librosa.load('short.wav',sr=SR)

size_frame= 512 #2のべき乗

hamming_window =np.hamming(size_frame)

 #シフトサイズ
size_shift= 16000/100 #0.01秒(10 msec)

 #スペクトログラムを保存するlist
spectrogram=[]

for i in np.arange(0, len(x)-size_frame,size_shift):

    idx= int(i) # arangeのインデクスはfloatなのでintに変換
    x_frame=x[idx:idx+size_frame]

    # np.fft.rfftを使用するとFFTの前半部分のみが得られる
    fft_spec=np.fft.rfft(x_frame * hamming_window)

    #複素スペクトログラムを対数振幅スペクトログラムに
    fft_log_abs_spec=np.log(np.abs(fft_spec))

    #計算した対数振幅スペクトログラムを配列に保存
    spectrogram.append(fft_log_abs_spec)
    
def zero_cross(waveform):
    
    zc=0
    
    for i in range(len(waveform)-1):
        if ( (waveform[i]>0.0 and waveform[i+1] < 0.0) or (waveform[i]<0.0 and waveform[i+1] > 0.0)):
            zc+=1
    return zc
#音声波形データを受け取り，ゼロ交差数を計算する関数
def zero_cross_short(waveform):
    d=np.array(waveform)
    return sum([1 if x<0.0 else 0 for x in d[1:] * d[:-1]])