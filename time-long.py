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

time_axis = np.arange(len(db)) * (size_shift / SR)
start_point = 0
i = 0
while i < len(db):
    if db[i] > -28:
        start_point = i
        print("音声開始位置 (秒):", time_axis[i])
        break
    i += 1
    
start_point += int(5 * (SR / size_shift))  # 5秒分シフト

while start_point < len(db):
    if db[start_point] < -28:
        print("音声終了位置 (秒):", time_axis[start_point])
        break
    start_point += 1
    
print('処理終了')
    

