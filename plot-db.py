 #ライブラリの読み込み
import matplotlib.pyplot as plt
import numpy as np
import librosa

#サンプリングレート
SR= 16000

#音声ファイルの読み込み
x,_ =librosa.load('long.wav',sr=SR)

 #フレームサイズ
size_frame= 512 #2のべき乗

 #フレームサイズに合わせてハミング窓を作成
hamming_window =np.hamming(size_frame)

 #シフトサイズ
size_shift= 16000/100 #0.01秒(10 msec)

 #dbを保存するlist
db=[]

for i in np.arange(0, len(x)-size_frame,size_shift):

#該当フレームのデータを取得
    idx= int(i) # arangeのインデクスはfloatなのでintに変換
    x_frame=x[idx:idx+size_frame]
    
    current_rms = np.sqrt(np.mean(x_frame**2))
    current_db = 20 * np.log10(current_rms)

    db.append(current_db)


#画像として保存するための設定
fig=plt.figure()

#dbを描画
plt.xlabel('time[s]') # x軸のラベルを設定
plt.ylabel(' dB ') # y軸のラベルを設定
time_axis = np.arange(len(db)) * (size_shift / SR)
plt.plot(time_axis,db)


plt.show()

fig.savefig('plot-db-long.png')