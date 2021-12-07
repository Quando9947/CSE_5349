import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import mean
from scipy.fftpack import fft, fftshift
import math
import os

def string2array(input_str):
    line =  ( input_str.strip('\n,],[').split(',') ) 
    line = np.array([ eval(num) for num in line])
    return line


files = os.listdir('./')

for file in files:
    if (not os.path.isdir(file) and file.split('.')[1] == 'txt' and file.split('.')[0] =='new_test'):
        print('file name is', file)
        data = open('./'+file)
        lines = data.readlines()
        valid_values = []
        for line in lines:
            #print(line)
            if(line[0:21] == 'Load_cell output val:'):
                valid_values.append( abs(float(line[21:].strip('\n')) ) )

        #print('valid_values are\n', valid_values)
        print('length of valid_values is', len(valid_values))
            

        fig = plt.figure()

        ax1 = fig.add_subplot(111)
        ax1.plot(valid_values)
        ax1.set_title('Whole signal')
        plt.show()

        falling = valid_values[240:340]  
        #falling.txt 300-400, 400-480, 480:575, 575:670, 675:775, 780:880, 1080:1180, 1190:1290, 1290:1390, 
        #1520:1620, 1620:1710, 1710:1800, 1800:1900, 1900:2000
        #falling2.txt, 200:270, 270:340, 340:440, 440:510，510:610, 610:690, 690:780, 780:880, 880:980, 980:1070, 1070:1150, 1150:1230,
        #1230:1330, 1340:1440, 1510:1580, 1580:1650,1650:1730,1730:1800,1800:1870,1870:1950,1950:2030,2030:2110,2110:2170,2170:2230,
        #2170:2230, 2230:2290, 2290:2345, 2345:2405, 2405:2460, 2460:2520, 2530:2590,2590:2670,2670:2735,2735:2800,2800:2870,2870:2940,
        #2940:3000,3000:3060,3070:3130,3140:3200, 3210:3300
        print(type(falling))
        for i in range( 100 - len(falling)):
            falling.append(0)
        plt.plot(falling)
        plt.show()

        if (1):
            for k in range(0,len(falling)+5-100,100):
                print(k)
                f = open('./new_testing.txt','w')
                print( str(falling[k:k+100]) )
                f.write(str(falling[k:k+100]))
                f.write('\n')
                f.close()
    
    '''
    elif(not os.path.isdir(file) and file.split('.')[1] == 'txt' and file.split('.')[0] =='standsit_dataset'):
        data = open('./'+file)
        lines = data.readlines()
        fig = plt.figure()
        for line in lines:
            line = string2array(line)
            print('line length is',len(line))
            
            plt.plot(line)
            plt.title('line')
            plt.show()

    '''
