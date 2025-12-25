import matplotlib.pyplot as plt
import numpy as np
import librosa

SR = 16000

# スペクトル包絡を決定する係数
spectral_envelope_number = 13 

size_frame= 512 #2のべき乗

size_shift= 16000/100 #0.01秒(10 msec)

hamming_window =np.hamming(size_frame)

spectrogram=[]


# x_l, _ = librosa.load('long.wav', sr=SR)
x_l, _ = librosa.load('kimi.wav', sr=SR)
x_s, _ = librosa.load('short.wav', sr=SR)

x_a = x_s[int(SR * 0.5):int(SR * 1.7)]
x_i = x_s[int(SR * 1.7): int(SR * 2.5)]
x_u = x_s[int(SR * 2.5): int(SR * 4)]
x_e = x_s[int(SR * 4): int(SR * 5)]
x_o = x_s[int(SR * 5): int(SR * 6)]

def cepstrum(amplitude_spectrum):  
    amplitude_spectrum += 1e-6
    log_spectrum=np.log(amplitude_spectrum)
    cepstrum=np.fft.fft(log_spectrum)
    return cepstrum

def calculate_cepstrum_real(x):
    fft_spec = np.fft.fft(x)
    fft_abs_spec = np.abs(fft_spec)
    ceps_result = cepstrum(fft_abs_spec) 
    ceps_result_real = np.real(ceps_result)
    return ceps_result_real

def calculate_params(cepstrum_data):
    average = np.mean(cepstrum_data, axis=0)
    variance = np.var(cepstrum_data, axis=0)
    return average, variance

all_ceps_a = []
for i in np.arange(0, len(x_a)-size_frame,size_shift):
    
    idx= int(i) 
    x_frame= x_a[idx:idx+size_frame]
    
    cepstrum_real = calculate_cepstrum_real(x_frame)
    
    ceps_windowed = np.zeros(spectral_envelope_number, dtype=complex) 

    ceps_windowed[0:spectral_envelope_number] = cepstrum_real[0:spectral_envelope_number] 
    
    all_ceps_a.append(ceps_windowed.real) 
    
all_ceps_i = []
for i in np.arange(0, len(x_i)-size_frame,size_shift):
    
    idx= int(i) 
    x_frame= x_i[idx:idx+size_frame]
    
    cepstrum_real = calculate_cepstrum_real(x_frame)
    
    ceps_windowed = np.zeros(spectral_envelope_number, dtype=complex) 

    ceps_windowed[0:spectral_envelope_number] = cepstrum_real[0:spectral_envelope_number] 
    
    all_ceps_i.append(ceps_windowed.real) 
    
all_ceps_u = []
for i in np.arange(0, len(x_u)-size_frame,size_shift):
    
    idx= int(i) 
    x_frame= x_u[idx:idx+size_frame]
    
    cepstrum_real = calculate_cepstrum_real(x_frame)
    
    ceps_windowed = np.zeros(spectral_envelope_number, dtype=complex) 

    ceps_windowed[0:spectral_envelope_number] = cepstrum_real[0:spectral_envelope_number] 
    
    all_ceps_u.append(ceps_windowed.real) 
    
all_ceps_e = []
for i in np.arange(0, len(x_e)-size_frame,size_shift):
    
    idx= int(i) 
    x_frame= x_e[idx:idx+size_frame]
    
    cepstrum_real = calculate_cepstrum_real(x_frame)
    
    ceps_windowed = np.zeros(spectral_envelope_number, dtype=complex) 

    ceps_windowed[0:spectral_envelope_number] = cepstrum_real[0:spectral_envelope_number] 
        
    all_ceps_e.append(ceps_windowed.real) 
    
all_ceps_o = []
for i in np.arange(0, len(x_o)-size_frame,size_shift):
    
    idx= int(i) 
    x_frame= x_o[idx:idx+size_frame]

    cepstrum_real = calculate_cepstrum_real(x_frame)
    
    ceps_windowed = np.zeros(spectral_envelope_number, dtype=complex) 

    ceps_windowed[0:spectral_envelope_number] = cepstrum_real[0:spectral_envelope_number] 
    
    all_ceps_o.append(ceps_windowed.real) 
    
model_a = calculate_params(np.array(all_ceps_a))
model_i = calculate_params(np.array(all_ceps_i))
model_u = calculate_params(np.array(all_ceps_u))
model_e = calculate_params(np.array(all_ceps_e))
model_o = calculate_params(np.array(all_ceps_o))

def calculate_log_likelihood(cepstrum_feature, average, variance):
    D = len(cepstrum_feature)
    variance_safe = variance + 1e-9
    term1 = - (D / 2.0) * np.log(2 * np.pi)
    term2 = - 0.5 * np.sum(np.log(variance_safe))
    diff = cepstrum_feature - average
    term3 = - 0.5 * np.sum((diff ** 2) / variance_safe)
    log_likelihood = term1 + term2 + term3
    return log_likelihood

likelihoods = []

def plot_likelihoods(x_frame, model_a, model_i, model_u, model_e, model_o, likelihoods):
    
    cepstrum_real = calculate_cepstrum_real(x_frame) 

    ceps_windowed = np.zeros(spectral_envelope_number, dtype=complex) 

    ceps_windowed[0:spectral_envelope_number] = cepstrum_real[0:spectral_envelope_number] 

    cepstrum_feature = ceps_windowed.real 

    predict_a = calculate_log_likelihood(cepstrum_feature, model_a[0], model_a[1])
    predict_i = calculate_log_likelihood(cepstrum_feature, model_i[0], model_i[1])
    predict_u = calculate_log_likelihood(cepstrum_feature, model_u[0], model_u[1])
    predict_e = calculate_log_likelihood(cepstrum_feature, model_e[0], model_e[1])
    predict_o = calculate_log_likelihood(cepstrum_feature, model_o[0], model_o[1])

    predict_result = np.argmax([predict_a, predict_i, predict_u, predict_e, predict_o])
    
    likelihoods.append(predict_result)
    
    return likelihoods
    
for i in np.arange(0, len(x_l)-size_frame,size_shift):
    
    idx= int(i) 
    x_frame= x_l[idx:idx+size_frame]
    
    fft_spec=np.fft.rfft(x_frame * hamming_window)
    fft_log_abs_spec=np.log(np.abs(fft_spec))
    spectrogram.append(fft_log_abs_spec)
    
    likelihoods = plot_likelihoods(x_frame, model_a, model_i, model_u, model_e, model_o, likelihoods)
    
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

time_end = (len(spectrogram) * size_shift) / SR 

ax1.set_ylabel('Frequency [Hz]') 
ax1.imshow(
    np.flipud(np.array(spectrogram).T), 
    extent=[0, time_end, 0, SR / 2],
    aspect='auto',
    interpolation='nearest'
)
ax1.set_ylim(0, 500) 

time_points = np.linspace(0, time_end, len(likelihoods))

ax2.plot(time_points, likelihoods, drawstyle='steps-post', color='red', linewidth=2)
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Predicted Vowel Index')
ax2.set_yticks(np.arange(5)) 
ax2.set_yticklabels(['a (0)', 'i (1)', 'u (2)', 'e (3)', 'o (4)'])
ax2.set_ylim(-0.5, 4.5) 
ax2.grid(axis='y', linestyle='--')

plt.tight_layout() 
plt.show()
