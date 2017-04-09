import numpy as np;
import matplotlib.pyplot as plt;
from scipy.signal import butter, lfilter
#from scipy.signal import  spectrogram
#import scipy as sp
from scipy.io import wavfile 
from scipy.io.wavfile import write
from matplotlib.mlab import specgram;
from matplotlib.mlab import window_hanning;



def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y



# Filter requirements.
order = 6
fs = 44100.0       # sample rate, Hz
cutoff = 1000.667  # desired cutoff frequency of the filter, Hz

# Get the filter coefficients so we can check its frequency response.
b, a = butter_lowpass(cutoff, fs, order)


fs = 44100;
wav_arr = wavfile.read('sentence.wav');
#fft_ip_arr = wav_arr[1][8192*16:8192*17 -1, 0];
print wav_arr[1][:,0].shape
print wav_arr[0]
print wav_arr[1][:100, 0].shape
#print ();
#print fft_op.shape
#plt.plot(np.arange(44100), np.absolute(np.fft.fft(wav_arr[1][:4096, 0], 44100)))
#plt.show()
print "mean:", np.mean(abs(wav_arr[1][:43000, 0]));
print "median", np.median(abs(wav_arr[1][:43000, 0]));

tmp_arr  = [i if i > 350 else 0 for i in wav_arr[1][:, 0]];
write('test.wav', 44100, np.int16(tmp_arr));


filter_op = butter_lowpass_filter(wav_arr[1][:, 0], 1000, 44100, 6);
NFFT = int(fs * 0.1);
noverlap = int(NFFT/4);
print NFFT, noverlap;
print "spec";
Pxx, freqs, t =specgram(wav_arr[1][:43000,0], NFFT=NFFT, window=window_hanning, noverlap=noverlap);
print Pxx, freqs, t;
plt.pcolormesh(t, freqs, Pxx);
plt.colorbar();
plt.show();


Pxx, freqs, t =specgram(filter_op, NFFT=NFFT, window=window_hanning, noverlap=noverlap);
print Pxx, freqs, t;
plt.pcolormesh(t, freqs, Pxx);
plt.colorbar();
plt.show();
#print np.amax(butter_lowpass_filter(wav_arr[1][:, 0], 1000, 44100, 6));
#scaled = np.int16(butter_lowpass_filter(wav_arr[1][:, 0], 500, 44100, 6)) 
#write('test.wav', 44100, scaled)
#print scaled.shape
#plt.plot(np.arange(44100), np.absolute(np.fft.fft(scaled, 44100)))
#plt.show()

#fop, t, sxx, va = plt.specgram(wav_arr[1][:,0], 44100);
#plt.pcolormesh(t, fop, sxx);
#plt.show();

#fft_ip_arr = wav_arr[1][8192*16:8192*17 -1, 0];
#fft_op = np.fft.fft(fft_ip_arr);
#np.absolute(fft_op).shape
#print np.absolute(fft_op).shape
#print np.arange(8191).reshape(1, 8191).shape
#plt.plot(np.arange(8191), np.absolute(fft_op)[:8191])
#plt.show()
