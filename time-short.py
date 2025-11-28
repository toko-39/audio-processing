 #ライブラリの読み込み
import matplotlib.pyplot as plt
import numpy as np
import librosa

#サンプリングレート
SR= 16000

#音声ファイルの読み込み
x,_ =librosa.load('short.wav',sr=SR)

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
        start_point_a = i
        print("音声開始位置 あ (秒):", time_axis[i])
        break
    i += 1
    
while start_point_a < len(db):
    if db[start_point_a] < -28:
        end_point_a = start_point_a
        print("音声終了位置 あ (秒):", time_axis[start_point_a])
        break
    start_point_a += 1

while end_point_a < len(db):
    if db[end_point_a] > -28:
        start_point_i = end_point_a
        print("音声開始位置 い (秒):", time_axis[end_point_a])
        break
    end_point_a += 1
    
while start_point_i < len(db):
    if db[start_point_i] < -28:
        end_point_i = start_point_i
        print("音声終了位置 い (秒):", time_axis[start_point_i])
        break
    start_point_i += 1

while end_point_i < len(db):
    if db[end_point_i] > -28:
        start_point_u = end_point_i
        print("音声開始位置 う (秒):", time_axis[end_point_i])
        break
    end_point_i += 1

while start_point_u < len(db):
    if db[start_point_u] < -28:
        end_point_u = start_point_u
        print("音声終了位置 う (秒):", time_axis[start_point_u])
        break
    start_point_u += 1

while end_point_u < len(db):
    if db[end_point_u] > -28:
        start_point_e = end_point_u
        print("音声開始位置 え (秒):", time_axis[end_point_u])
        break
    end_point_u += 1

while start_point_e < len(db):
    if db[start_point_e] < -28:
        end_point_e = start_point_e
        print("音声終了位置 え (秒):", time_axis[start_point_e])
        break
    start_point_e += 1

while end_point_e < len(db):
    if db[end_point_e] > -28:
        start_point_o = end_point_e
        print("音声開始位置 お (秒):", time_axis[end_point_e])
        break
    end_point_e += 1

while start_point_o < len(db):
    if db[start_point_o] < -28:
        end_point_o = start_point_o
        print("音声終了位置 お (秒):", time_axis[start_point_o])
        break
    start_point_o += 1

print('処理終了')
    

