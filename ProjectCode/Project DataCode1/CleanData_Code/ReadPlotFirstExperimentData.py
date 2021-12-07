import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import mean
from scipy.fftpack import fft, fftshift
import math

data = open('./FirstExperimentData.txt')
lines = data.readlines()

valid_values = []
for line in lines:
    print(line)
    if(line[0:21] == 'Load_cell output val:'):
        valid_values.append( float(line[21:].strip('\n')) )

print('valid_values are\n', valid_values)
print('length of valid_values is', len(valid_values))


walking = valid_values[215:270] - mean( np.array(valid_values[215:270]) )
jumping = valid_values[335:395] - mean(np.array(valid_values[335:395]) )
fall = valid_values[460:530]- mean(np.array(valid_values[460:530]))

fig = plt.figure()

ax1 = fig.add_subplot(221)
ax1.plot(valid_values)
ax1.set_title('Whole signal')

ax2_mean = np.mean(walking)
ax2_std = np.std(walking)
ax2 = fig.add_subplot(222)
ax2.plot(walking)
ax2.set_title('walking')

ax3_mean = np.mean(jumping)
ax3_std = np.std(jumping)
ax3 = fig.add_subplot(223)
ax3.plot(jumping)
ax3.set_title('jumping')


ax4_mean = np.mean(fall)
ax4_std = np.std(fall)
ax4 = fig.add_subplot(224)
ax4.plot(fall)
ax4.set_title('fall down')

print('ax2_mean {}, ax2_std {}, ax3_mean {}, ax3_std {}, ax4_mean {}, ax4_std {}'.format(ax2_mean,ax2_std, ax3_mean,ax3_std, ax4_mean, ax4_std))
plt.show()

#fft        
fft_all = fft(valid_values)
abs_all= fftshift( np.abs(fft_all) / len(fft_all)       )
abs_all_len = len(abs_all) 
x = np.arange((-int(abs_all_len/2)),math.ceil(abs_all_len/2),1)
fig2 = plt.figure()
ax1 = fig2.add_subplot(221)
ax1.plot( x,abs_all)   
ax1.set_title('Whole Signal FFT')

fft_walking = fftshift(  fft(walking) )
abs_walking= np.abs(fft_walking) / len(fft_walking)       
abs_walking_len = len(abs_walking) 
x = np.arange((-int(abs_walking_len/2)),math.ceil(abs_walking_len/2),1)
ax2 = fig2.add_subplot(222)
ax2.plot( x,abs_walking)   
ax2.set_title('Walking FFT')

fft_jumping = fftshift (fft(jumping))
abs_jumping=  np.abs(fft_jumping) / len(fft_jumping)       
abs_jumping_len = len(abs_jumping) 
x = np.arange((-int(abs_jumping_len/2)),math.ceil(abs_jumping_len/2),1)
ax3 = fig2.add_subplot(223)
ax3.plot( x,abs_jumping)   
ax3.set_title('jumping FFT')

fft_fall = fft(fall)
abs_fall= fftshift( np.abs(fft_fall) / len(fft_fall)       )
abs_fall_len = len(abs_fall) 
x = np.arange((-int(abs_fall_len/2)),math.ceil(abs_fall_len/2),1)
ax4 = fig2.add_subplot(224)
ax4.plot( x,abs_fall)   
ax4.set_title('fall down FFT')

plt.show()