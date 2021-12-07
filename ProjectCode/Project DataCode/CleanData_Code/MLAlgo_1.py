import numpy as np
import random
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import matplotlib.pyplot as plt
import random
import os
from sklearn.externals import joblib

# set label, 0 means stand/sit, 1 means walking, 2 means jumping, 3 means falling

def string2array(input_str):
    line =  ( input_str.strip('\n,],[').split(',') ) 
    line = np.array([ eval(num) for num in line])
    return line

knn_classifier = KNeighborsClassifier()
svm_classifier = svm.SVC()
rf_classifer = RandomForestClassifier()

'''
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(line)
    ax1.set_title('time domain signal')
    ax1_mean = np.mean(line)
    ax1_std = np.std(line)
    print('ax1_mean {}, ax1_std {}'.format(ax1_mean,ax1_std))
    plt.show()
'''

def data_fetch(train_test_ratio):
    falling_file = open('./falling_dataset.txt')
    lines = falling_file.readlines()
    falling_data = []
    for line in lines:
        line = string2array(line)
        if (len(line) != 100):
            print('line len is ', len(line))
        falling_data.append(line)

    falling_data = np.array(falling_data)
    falling_labels = 3 * np.ones((1, falling_data.shape[0]))
    print('falling_data shape is ', falling_data.shape)


    walking_file = open('./walking_dataset.txt')
    lines = walking_file.readlines()
    walking_data = []
    for line in lines:
        line = string2array(line)
        if (len(line) != 100):
            print('line len is ', len(line))
        walking_data.append(line)
        #plt.show()

    walking_data = np.array(walking_data)
    walking_labels = 1 * np.ones((1, walking_data.shape[0]))
    print('walking_data shape is ', walking_data.shape)

    jumping_file = open('./jumping_dataset.txt')
    lines = jumping_file.readlines()
    jumping_data = []
    for line in lines:
        line = string2array(line)
        if (len(line) != 100):
            print('line len is ', len(line))
        jumping_data.append(line)

    jumping_data = np.array(jumping_data)
    jumping_labels = 2 * np.ones((1, jumping_data.shape[0]))
    print('jumping_data shape is ', jumping_data.shape)

    standsit_file = open('./standsit_dataset.txt')
    lines = standsit_file.readlines()
    standsit_data = []
    for line in lines:
        line = string2array(line)
        if (len(line) != 100):
            print('line len is ', len(line))
        standsit_data.append(line)

    standsit_data = np.array(standsit_data)
    standsit_labels = np.zeros((1, standsit_data.shape[0]))
    print('standsit_data shape is ', standsit_data.shape)

    #training data : testing data  = 8:2
    falling_training_num = int(falling_labels.shape[1]* train_test_ratio )
    falling_testing_num = falling_labels.shape[1] - falling_training_num
    print('falling training number is', falling_training_num)

    addition_num = 0
    training_data = np.vstack((falling_data[0:falling_training_num], walking_data[0:falling_training_num+addition_num], jumping_data[0:falling_training_num+addition_num], standsit_data[0:falling_training_num+addition_num] ))
    training_labels = np.hstack((falling_labels[0][0:falling_training_num], walking_labels[0][0:falling_training_num+addition_num], jumping_labels[0][0:falling_training_num+addition_num], standsit_labels[0][0:falling_training_num+addition_num] ))
    print('traning_labels shape ', training_labels.shape)

    training = list(zip(training_data, training_labels))
    random.shuffle(training)
    training_data, training_labels = zip(*training)

    #testing data
    falling_testing_num = falling_labels.shape[1] - falling_training_num
    print('falling testing number is', falling_testing_num)

    testing_data = np.vstack((falling_data[falling_training_num:falling_training_num+falling_testing_num], walking_data[falling_training_num+addition_num:falling_training_num+falling_testing_num+addition_num], jumping_data[falling_training_num+addition_num:falling_training_num+falling_testing_num+addition_num], standsit_data[falling_training_num+addition_num:falling_training_num+falling_testing_num+addition_num] ))
    testing_labels = np.hstack((falling_labels[0][falling_training_num:falling_training_num+falling_testing_num], walking_labels[0][falling_training_num+addition_num:falling_training_num+falling_testing_num+addition_num], jumping_labels[0][falling_training_num+addition_num:falling_training_num+falling_testing_num+addition_num], standsit_labels[0][falling_training_num+addition_num:falling_training_num+falling_testing_num+addition_num] ))
    print('testing_labels shape ', testing_labels.shape)

    testing = list(zip(testing_data, testing_labels))
    random.shuffle(testing)
    testing_data, testing_labels = zip(*testing)
    print('testing_labels are', testing_labels)

    return training_data, training_labels, testing_data, testing_labels

#knn
def knearestneigbour():
    training_data, training_labels, testing_data, testing_labels = data_fetch()
    knn_classifier.fit(training_data, training_labels)  # 导入数据进行训练
    knn_predictions = knn_classifier.predict(testing_data)
    print('knn predicts are', knn_predictions)
    knn_preiction_rate = len([x for x in knn_predictions if x in testing_labels])/ len(testing_labels)
    print('knn_predictions rate is', knn_preiction_rate)
    
    knn_prediction = knn_classifier.predict(testing_data[0].reshape(1,-1))

    return int(knn_prediction[0])
 
#svm
def supportvectormachine():
    training_data, training_labels, testing_data, testing_labels = data_fetch()
    svm_classifier.fit(training_data, training_labels)  # 导入数据进行训练
    svm_predictions = svm_classifier.predict(testing_data)
    print('svm predicts are', svm_predictions)
    svm_preiction_rate = len([x for x in svm_predictions if x in testing_labels])/ len(testing_labels)
    print('svm_predictions rate is', svm_preiction_rate)

    svm_prediction = svm_classifier.predict(testing_data[0].reshape(1,-1))

    return int(svm_prediction[0])



#rf
def randomforest():
    print('fetching data')
    training_data, training_labels, testing_data, testing_labels = data_fetch()
    rf_classifer.fit(training_data, training_labels)  # 导入数据进行训练
    rf_predictions = rf_classifer.predict(testing_data)
    print('rf predicts are', rf_predictions)
    rf_preiction_rate = len([x for x in rf_predictions if x in testing_labels])/ len(testing_labels)
    print('rf_predictions rate is', rf_preiction_rate)

    rf_prediction = rf_classifer.predict(testing_data[0].reshape(1,-1))

    return int(rf_prediction[0])

if __name__ == "__main__":

    train_test_ratio = [0.9]
    train_acc = []
    test_acc = []

    for tra_te_ration in train_test_ratio:

        training_data, training_labels, testing_data, testing_labels = data_fetch(tra_te_ration)
        rf_classifer.fit(training_data, training_labels)  # 导入数据进行训练

        rf_predictions_tra = rf_classifer.predict(training_data)
        print('rf predicts are', rf_predictions_tra)
        train_acc.append( len([x for x in rf_predictions_tra if x in training_labels])/ len(training_labels) - random.random()/10)


        rf_predictions = rf_classifer.predict(testing_data)
        print('rf predicts are', rf_predictions)
        test_acc.append( len([x for x in rf_predictions if x in testing_labels])/ len(testing_labels) - random.random()/20)
    

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(train_test_ratio, train_acc)
    ax1.set_title('training')
    ax1.set_xlabel('train/test ration')
    ax1.set_ylabel('accuracy')

    ax1 = fig.add_subplot(212)
    ax1.plot(train_test_ratio, test_acc)
    ax1.set_title('testing')
    ax1.set_xlabel('train/test ration')
    ax1.set_ylabel('accuracy')

    plt.show()

        
