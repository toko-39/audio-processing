#計算機科学実験及演習4「音響信号処理」
# サンプルソースコード

#音声ファイルを読み込み，波形を図示する．

#ライブラリの読み込み
import matplotlib.pyplot as plt
import librosa

 #サンプリングレート
SR= 16000

 #音声ファイルの読み込み
x,_ =librosa.load('short.wav',sr=SR)

# xに波形データが保存される
#第二戻り値はサンプリングレート（ここでは必要ないので_としている）
#波形データを標準出力して確認
print(x)


#波形を画像に表示・保存


#画像として保存するための設定
#画像サイズを1000 x 400に設定
fig=plt.figure(figsize=(10,4))

#波形を描画
plt.plot(x) #描画データを追加
plt.xlabel('Sampling point') # x軸のラベルを設定
plt.show() #表示
 #画像ファイルに保存
fig.savefig('plot-waveform.png')