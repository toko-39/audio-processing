import matplotlib.pyplot as plt
import numpy as np
import librosa

SR = 16000

# スペクトル包絡を決定する係数
M_COEFFICIENTS = 13 

x, _ = librosa.load('long.wav', sr=SR)
x_s, _ = librosa.load('short.wav', sr=SR)

x_a = x_s[int(SR * 0.5):int(SR * 1.7)]
x_i = x_s[int(SR * 1.7): int(SR * 2.5)]
x_u = x_s[int(SR * 2.5): int(SR * 4)]
x_e = x_s[int(SR * 4): int(SR * 5)]
x_o = x_s[int(SR * 5): int(SR * 6)]

def cepstrum(amplitude_spectrum):   
    log_spectrum=np.log(amplitude_spectrum)
    cepstrum=np.fft.fft(log_spectrum)
    return cepstrum

def spectral_envelope_and_original_spectrum(x, M_COEFFICIENTS):
    
    fft_spec = np.fft.rfft(x)
    
    fft_abs_spec = np.abs(fft_spec)
    
    ceps_result = cepstrum(fft_abs_spec)

    N_spec = len(ceps_result) # ケプストラムの結果の長さ

    ceps_windowed = np.zeros(N_spec, dtype=complex) # 複素数結果に対応

    ceps_windowed[0:M_COEFFICIENTS] = ceps_result[0:M_COEFFICIENTS] # 低ケプストラム係数を保持

    log_envelope_full = np.fft.ifft(ceps_windowed) # 逆フーリエ変換で対数スペクトル包絡を取得

    log_envelope = np.real(log_envelope_full) # 実数部分を抽出

    log_spectrum_original = np.log(fft_abs_spec) # 元の対数スペクトル

    freq_axis = np.linspace(0, SR / 2, N_spec)
    
    return freq_axis, log_envelope, log_spectrum_original

freq_axis, log_envelope, log_spectrum_original = spectral_envelope_and_original_spectrum(x, M_COEFFICIENTS)

plt.plot(freq_axis, log_spectrum_original, label='Original Log Spectrum', color='gray', alpha=0.6)
plt.plot(freq_axis, log_envelope, label=f'Spectral Envelope (M={M_COEFFICIENTS})', color='red', linewidth=2)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Log Amplitude')
plt.grid(True)
plt.show()
