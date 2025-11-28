
#計算機科学実験及演習4「音響信号処理」


#音声ファイルを読み込み，フーリエ変換を行う．

#ライブラリの読み込み
import matplotlib.pyplot as plt
import numpy as np
import librosa

 #配列aのindex番目の要素がピーク（両隣よりも大きい）であればTrueを返す
def is_peak(a,index):
    if index <=0 or index >= len (a)-1:
        return False
    if a[index] > a[index -1] and a[index] > a[index +1]:
        return True

 #サンプリングレート
SR= 16000

fundamental_frequency = []
 #音声ファイルの読み込み
x,_ =librosa.load('short.wav',sr=SR)

# x_a = x[int(SR * 0.5):int(SR * 1.7)]
# x_i = x[int(SR * 1.7): int(SR * 2.5)]
# x_u = x[int(SR * 2.5): int(SR * 4)]
# x_e = x[int(SR * 4): int(SR * 5)]
# x_o = x[int(SR * 5): int(SR * 6)]


 #フレームサイズ
size_frame= 512 #2のべき乗

 #フレームサイズに合わせてハミング窓を作成
hamming_window =np.hamming(size_frame)

 #シフトサイズ
size_shift= 16000/100 #0.01秒(10 msec)
    
def calculate_fundamental_frequency(x):
 #自己相関が格納された，長さがlen(x)*2-1の対称な配列を得る
    autocorr=np.correlate(x,x, 'full')

 #不要な前半を捨てる
    autocorr=autocorr[len (autocorr)//2: ]

 #ピークのインデックスを抽出する
    peakindices=[i for i in range (len (autocorr)) if is_peak (autocorr,i)]

 #インデックス0がピークに含まれていれば捨てる
    peakindices=[i for i in peakindices if i !=0]
    
# ピークが一つも見つからなかった場合は0を返す
    if not peakindices:
        return 0.0

 #自己相関が最大となるインデックスを得る
    max_peak_index = max(peakindices,key=lambda index:autocorr [index])

    #インデックスに対応する周波数を計算する
    fundamental_frequency = SR / max_peak_index
    
    return fundamental_frequency

for i in np.arange(0, len(x)-size_frame,size_shift):
    
    idx= int(i) 
    x_frame=x[idx:idx+size_frame]
    # x_windowed = x_frame * hamming_window
    fundamental_frequency.append(calculate_fundamental_frequency(x_frame))
    
fig=plt.figure()
time_axis = np.arange(len(fundamental_frequency)) * (size_shift / SR)
plt.ylabel('frequency [Hz]')
plt.xlabel('time[s]') # x軸のラベルを設定
plt.xlim([0, time_axis[-1]])
plt.ylim([0, 500]) #縦軸の表示範囲を0〜500 Hzに制限
plt.plot(time_axis,fundamental_frequency)

 #表示
plt.show()
    
