import matplotlib.pyplot as plt
import numpy as np
import librosa

SR= 16000

x,_ =librosa.load('ai.wav',sr=SR)

size_frame= 512 #2のべき乗

hamming_window =np.hamming(size_frame)

 #シフトサイズ
size_shift= 16000/100 #0.01秒(10 msec)

"""スペクトログラムの計算"""

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

"""自己相関、基本周波数の計算"""
#配列aのindex番目の要素がピーク（両隣よりも大きい）であればTrueを返す
def is_peak(a,index):
    if index <=0 or index >= len (a)-1:
        return False
    if a[index] > a[index -1] and a[index] > a[index +1]:
        return True

#音声波形データを受け取り，ゼロ交差数を計算する関数
def zero_cross(waveform):
    
    zc=0
    
    for i in range(len(waveform)-1):
        if ( (waveform[i]>0.0 and waveform[i+1] < 0.0) or (waveform[i]<0.0 and waveform[i+1] > 0.0)):
            zc+=1
    return zc

fundamental_frequency = []

def calculate_fundamental_frequency(x):
    autocorr=np.correlate(x,x, 'full')

    autocorr=autocorr[len (autocorr)//2: ]

    peakindices=[i for i in range (len (autocorr)) if is_peak (autocorr,i)]

    peakindices=[i for i in peakindices if i !=0]
    
    if not peakindices:
        return 0.0

    max_peak_index = max(peakindices,key=lambda index:autocorr [index])

    fundamental_frequency = SR / max_peak_index
    
    zero_crossing = zero_cross(x)
    
    if zero_crossing * (SR / size_frame) > fundamental_frequency * 5:
        fundamental_frequency = 0.0
        
    current_rms = np.sqrt(np.mean(x_frame**2))
    current_db = 20 * np.log10(current_rms)
    
    if current_db < -40.0:
        fundamental_frequency = 0.0
        
    return fundamental_frequency

for i in np.arange(0, len(x)-size_frame,size_shift):
    
    idx= int(i) 
    x_frame=x[idx:idx+size_frame]
    fundamental_frequency.append(calculate_fundamental_frequency(x_frame))
    

zf = []
for i in np.arange(0, len(x)-size_frame,size_shift):
    
    idx= int(i) 
    x_frame=x[idx:idx+size_frame]
    zf.append(zero_cross(x_frame) * (SR / size_frame))

fig, axes = plt.subplots(2, 1, figsize=(8, 6))

axes[0].set_xlabel('sample') # x軸のラベルを設定
axes[0].set_ylabel('frequency [Hz]') # y軸のラベルを設定
axes[0].imshow(
np.flipud(np.array(spectrogram).T), #画像とみなすために，データを転地して上下反転
extent=[0, len(x),0,SR/2], # (横軸の原点の値，横軸の最大値，縦軸の原点の値，縦軸の最大値)
aspect='auto',
interpolation='nearest'
)
axes[0].set_ylim(0, 300) #縦軸の表示範囲を0〜300 Hzに制限


time_axis = np.arange(len(fundamental_frequency)) * (size_shift / SR)

axes[1].set_ylabel('frequency [Hz]')
axes[1].set_xlabel('time[s]') # x軸のラベルを設定
axes[1].set_xlim([0, time_axis[-1]])
axes[1].set_ylim([0, 500]) #縦軸の表示範囲を0〜500 Hzに制限
axes[1].plot(time_axis,fundamental_frequency)

# axes[1].set_ylabel('frequency [Hz]')
# axes[1].set_xlabel('time[s]') # x軸のラベルを設定
# axes[1].set_xlim([0, time_axis[-1]])
# axes[1].set_ylim([0, 500]) #縦軸の表示範囲を0〜500 Hzに制限
# axes[1].plot(time_axis, zf)


plt.tight_layout()
plt.show()