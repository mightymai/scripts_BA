import librosa
import librosa.display
import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import os

# path of audio file to be modified
file = '/home/maiwenn/Documents/1_Bachelorarbeit/mfcc/poppy_new.wav'

#bg_noise path
path = '/home/maiwenn/Documents/1_Bachelorarbeit/mfcc/_background_noise_/'

INT15_SCALE = np.power(2, 15)

# from CNN-CTC-LSTM implementation
def augment_audio(audio, bg_audios, target_length, fg_strech, bg_noise_prob,
                  bg_nsr):
  """Augments the foreground audio using time stretching, random padding
    and mixing with the background noise.
    audio.shape: [audio_length]
  """
  # time stretching
  if np.random.uniform(0, 1) < bg_noise_prob:
    audio_length = len(audio)
    step = np.random.uniform(1 / fg_strech, fg_strech)
    audio = np.interp(np.arange(0, audio_length, step),
                      np.arange(0, audio_length), audio)

  # fix length, random padding
  audio_length = len(audio)
  if audio_length <= target_length:
    pad_total = target_length - audio_length
    print('pad_total ' + str(pad_total))
    pad_bf = np.random.randint(0, pad_total + 1)
    print('pad_bf ' + str(pad_bf))
    fg_audio = np.pad(audio, [pad_bf, pad_total - pad_bf], mode='constant')
    print('fg_audio ' + str(fg_audio))
  else:
    start = (audio_length - target_length) // 2
    fg_audio = audio[start:start + target_length]

  # bg_noise_prob = 0: add no noise
  # bg_noise_prob != 0, bg_nsr = 0: only add noise to silence file
  if np.random.uniform(0, 1) < bg_noise_prob:
    fg_max = np.abs(fg_audio).max()
    if fg_max == 0:
      # the silence foreground audio
      volumn_max = 1
    else:
      volumn_max = fg_max * bg_nsr
    bg_noise = select_bg_audio(bg_audios, target_length, volumn_max)
  else:
    bg_noise = 0

  fg_audio += bg_noise
  fg_audio = np.clip(fg_audio, -1.0, 1.0)

  return fg_audio

def decode_audio(wav_file, target_length=-1):
  """Decodes audio wave file.
    audio_new.shape:[target_length]
  """
  _, audio = scipy.io.wavfile.read(wav_file)
  audio = audio.astype(np.float32, copy=False)
  # keep 0 not shifted. Not: (audio + 0.5) * 2 / (INT15_SCALE * 2 - 1)
  audio /= INT15_SCALE
  audio_length = len(audio)

  if target_length != -1:
    if audio_length < target_length:
      audio_new = np.zeros(target_length, dtype=np.float32)
      start = (target_length - audio_length) // 2
      print('first if ' + start)
      audio_new[start:start + audio_length] = audio
    else:
      start = (audio_length - target_length) // 2
      print('second if ' + start)
      audio_new = audio[start:start + target_length]
  else:
    print('else')
    audio_new = audio
  return audio_new

def select_bg_audio(bg_audios, target_length, volume_max=1):
  """Chooses a background noise from bg_audios list and returns a copy
    of the target length and scaled volume.
    bg_audios.type: list
  """
  index = np.random.randint(len(bg_audios))
  print(wav_files[index])
  print('index ' + str(index))
  audio = bg_audios[index]
  # randint: len(audio) - target_length + 1, high exclusive
  start = np.random.randint(0, len(audio) - target_length + 1)
  print('start ' + str(start))
  print('audio length ' + str(len(audio)))
  audio = audio[start:start + target_length]
  volume = np.random.uniform(0, volume_max)
  scaled = audio * volume
  return scaled

signal, sample_rate = librosa.load(file, sr=16000)
wav_signal = signal

# bg_noise file paths
wav_f = os.listdir(path)
wav_files = []
for wav in wav_f:
    wav = os.path.join(path, wav)
    wav_files.append(wav)
bg_audios = [decode_audio(w) for w in wav_files]

# from models.py and train.py
target_length = 1140
bg_noise_prob = 0.75
bg_nsr = 0.5
audio_length = int(sample_rate * target_length / 1000)
audio_pad = int((140 * sample_rate) / 1000)
fg_strech = audio_length / (audio_length - audio_pad)


audio = augment_audio(signal, bg_audios, audio_length, fg_strech, bg_noise_prob, bg_nsr)

plt.subplots(nrows=1, ncols=1)
librosa.display.waveplot(audio, sr=sample_rate, color='dimgray');
ax = plt.gca()
ax.invert_yaxis()
plt.axis('off')
plt.savefig('bg_noise_audio.png', dpi=400)

plt.subplots(nrows=1, ncols=1)
librosa.display.waveplot(wav_signal, sr=sample_rate, color='dimgray');
ax = plt.gca()
ax.invert_yaxis()
plt.axis('off')
plt.savefig('unprocessed_audio.png', dpi=400)



signal = audio
################################### Framing: 
# Cutting audio into small windows since changes in audio might be too big over time
# Typical frame sizes range from 20 ms to 40 ms with 50% (+/-10%) overlap between consecutive frames
frame_size = 0.025
frame_stride = 0.01 # 15ms overlap

# Convert from seconds to samples
frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  
#signal_length = len(emphasized_signal)
signal_length = len(signal)
frame_length = int(round(frame_length))
frame_step = int(round(frame_step))
# Make sure that we have at least 1 frame
num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  

# Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal
pad_signal_length = num_frames * frame_step + frame_length
z = np.zeros((pad_signal_length - signal_length))
#pad_signal = np.append(emphasized_signal, z) 
pad_signal = np.append(signal, z) 

indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
frames = pad_signal[indices.astype(np.int32, copy=False)]


################################### Applying window funktion (Hamming Window)
frames *= np.hamming(frame_length)


################################### Fourier Transform and Power Spectrum
# n-point fast fourier transform 
nfft = 256
# Magnitude of the FFT
mag_frames = np.absolute(np.fft.rfft(frames, nfft))
# Power Spectrum aka Periodogram  
pow_frames = ((1.0 / nfft) * ((mag_frames) ** 2))  

################################### Filter Banks
# number of filters
nfilt = 46
low_freq_mel = 20
# Convert Hz to Mel
high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  
# Equally spaced in Mel scale
mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  
# Convert Mel to Hz
hz_point = (700 * (10**(mel_points / 2595) - 1))  
#We don't have the frequency resolution required to put filters at the exact points calculated above, so we need to round those frequencies to the nearest FFT bin
bin = np.floor((nfft + 1) * hz_point / sample_rate)

fbank = np.zeros((nfilt, int(np.floor(nfft / 2 + 1))))
# Formel H_m(k)
for m in range(1, nfilt + 1):
    f_m_minus = int(bin[m - 1])   # left
    f_m = int(bin[m])             # center
    f_m_plus = int(bin[m + 1])    # right
    for k in range(f_m_minus, f_m):

    	fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
    for k in range(f_m, f_m_plus):
        fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

hz_points = np.floor(hz_point)
fbank_plot = np.zeros((nfilt, int(8000)))
fbank_plot.fill(np.nan)
# Formel H_m(k)
for m in range(1, nfilt + 1):
    f_m_minus = int(hz_points[m - 1])   # left
    f_m = int(hz_points[m])             # center
    f_m_plus = int(hz_points[m + 1])    # right
    for k in range(f_m_minus, f_m):
    	fbank_plot[m - 1, k] = (k - hz_points[m - 1]) / (hz_points[m] - hz_points[m - 1])
    for k in range(f_m, f_m_plus):
        fbank_plot[m - 1, k] = (hz_points[m + 1] - k) / (hz_points[m + 1] - hz_points[m])


plt.subplots(nrows=1, ncols=1, figsize=(8,1.8))
plt.plot(pow_frames)
plt.title('Power Spectrum')
plt.xlabel('Frequency (Hz)', fontsize=10)
plt.ylabel('Amplitude', fontsize=10)
plt.subplots_adjust(
top=0.875,
bottom=0.245,
left=0.08,
right=0.905,
hspace=0.2,
wspace=0.2)
plt.savefig('pow_frames.png', dpi=400)



plt.subplots(nrows=1, ncols=1, figsize=(8,1.8))
plt.plot(fbank_plot.T)
plt.title('filterbanks after triangular filter')
plt.xlabel('Frequency (Hz)', fontsize=10)
plt.ylabel('Amplitude', fontsize=10)
plt.yticks() 
#x = [1000, 2000, 4000, 8000]
#plt.xticks(x)
plt.subplots_adjust(
top=0.875,
bottom=0.245,
left=0.075,
right=0.905,
hspace=0.2,
wspace=0.2)
plt.savefig('filterbanks.png', dpi=400)


# filter_banks 
filter_banks = np.dot(pow_frames, fbank.T)
filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
filter_banks = 20 * np.log10(filter_banks)  # dB

################################### Apply DCT for decorrelation ==> MFCC
num_ceps = 12
mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13

# plotss
# filter_banks 
plt.subplots(nrows=1, ncols=1)
# Normalzation / Centering
filter_banks -= (np.mean(filter_banks,axis=0) + 1e-8)
plt.imshow(filter_banks, cmap='magma', aspect='equal')
#plt.xticks(np.arange(0, (filter_banks).shape[1], int((filter_banks).shape[1] / 4)))
#ax = plt.gca()
#plt..invert_yaxis()
#plt.title('Mel-scaled filter banks')
#plt.ylabel('Frequency (kHz)', fontsize=10)
#plt.xlabel('Time [ms]', fontsize=10)
#plt.subplots_adjust(
# top=0.875,
# bottom=0.245,
# left=0.075,
# right=0.905,
# hspace=0.2,
# wspace=0.2)
plt.axis('off')
#cbaxes = plt.axes([0.92, 0.2455, 0.03, 0.63]) 
#plt.colorbar(cax=cbaxes)
plt.savefig('filterbank_spectrogram.png', dpi=400)


# #MFCCs
# # Normalzation
plt.subplots(nrows=1, ncols=1)
mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
y1 = plt.imshow(mfcc.T, cmap='magma', aspect='equal')
# plt.xticks(np.arange(0, (mfcc.T).shape[1], int((mfcc.T).shape[1] / 4)))
ax = plt.gca()
ax.invert_yaxis()
# plt.title('MFCCs')
# plt.xlabel('Time [ms]', fontsize=10)
# plt.ylabel('MFCC Coeffcients', fontsize=10)
# plt.subplots_adjust(
# top=0.875,
# bottom=0.245,
# left=0.075,
# right=0.905,
# hspace=0.2,
# wspace=0.2)
plt.axis('off')
# cbaxes = plt.axes([0.92, 0.2455, 0.03, 0.63]) 
#plt.colorbar(cax=cbaxes)
plt.savefig('mfcc.png', dpi=400)
plt.show()


