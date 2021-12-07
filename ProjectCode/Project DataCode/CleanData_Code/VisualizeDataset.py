import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import mean
from scipy.fftpack import fft, fftshift
import math

def string2array(input_str):
    line =  ( input_str.strip('\n,],[').split(',') ) 
    line = np.array([ eval(num) for num in line])
    return line


data = open('./walking_dataset.txt')
lines = data.readlines()


for line in lines:
    line = string2array(line)
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(line)
    ax1.set_title('time domain signal')
    ax1_mean = np.mean(line)
    ax1_std = np.std(line)
    print('ax1_mean {}, ax1_std {}'.format(ax1_mean,ax1_std))

    #fft        
    line = line- mean( np.array(line) )
    fft_all = fft(line)
    abs_all= fftshift( np.abs(fft_all) / len(fft_all)       )
    abs_all_len = len(abs_all) 
    x = np.arange((-int(abs_all_len/2)),math.ceil(abs_all_len/2),1)
    ax2 = fig.add_subplot(212)
    ax2.plot( x,abs_all)   
    ax2.set_title('signal in frequency domain')

    plt.show()