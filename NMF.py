import numpy as np
import librosa
from sklearn.decomposition import NMF 
import matplotlib.pyplot as plt

SR = 16000
K = 5 # 分解するコンポーネント数

x, _ = librosa.load('kimi.wav', sr=SR)

D = librosa.stft(x)

Y = np.abs(D)

model = NMF(n_components=K, init='random', random_state=0, max_iter=500)

H = model.fit_transform(Y)
U = model.components_

eps = 1e-10

update_times = 500

for i in range(update_times):
    Y_hat = np.dot(H, U)
    
    H = H * (np.dot(Y, U.T) / (np.dot(Y_hat, U.T) + eps))
    
    Y_hat = np.dot(H, U)
    
    U = U * (np.dot(H.T, Y) / (np.dot(H.T, Y_hat) + eps))
    

plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
librosa.display.specshow(librosa.amplitude_to_db(Y, ref=np.max), 
                         y_axis='hz', x_axis='time', sr=SR)
plt.title('Y')

plt.subplot(2, 2, 2)
Y_hat = np.dot(H, U)
librosa.display.specshow(librosa.amplitude_to_db(Y_hat, ref=np.max), 
                         y_axis='hz', x_axis='time', sr=SR)
plt.title('HU')

plt.subplot(2, 2, 3)
# 周波数軸の値を生成
frequencies = librosa.fft_frequencies(sr=SR)
for k in range(K):
    plt.plot(frequencies, H[:, k], label=f'Component {k+1}')
plt.xlim(0, 1000)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.title('H')
plt.legend()

plt.subplot(2, 2, 4)
# 時間軸の値を生成
times = librosa.frames_to_time(np.arange(U.shape[1]), sr=SR)
for k in range(K):
    plt.plot(times, U[k, :], label=f'Component {k+1}')
plt.xlabel('Time [sec]')
plt.ylabel('Activation')
plt.title('U')
plt.legend()

plt.tight_layout()
plt.show()