#!/usr/bin/env python
# coding: utf-8



#checking what device is being used

import tensorflow as tf
print(tf.test.gpu_device_name())


# importing modules

import pandas as pd
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import seaborn as sns
from keras.models import Model
from keras.layers import Input, Dense
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from keras import backend as K
from keras import layers, losses
import itertools
import scipy
from scipy import io

#setting seaborn style
sns.set(style='whitegrid', context='notebook')

random_seed=4



# loading the mat file

# ********** loading 802.11ac file *****************
wifi_data_ac_16=scipy.io.loadmat('80211ac_5packets_16qam.mat')['waveStruct']['waveform'][0][0]
wifi_data_ac_16=wifi_data_ac_16.T
wifi_data_ac_16=wifi_data_ac_16.flatten()

wifi_data_ac_64=scipy.io.loadmat('80211ac_5packets.mat')['waveStruct']['waveform'][0][0]
wifi_data_ac_64=wifi_data_ac_64.T
wifi_data_ac_64=wifi_data_ac_64.flatten()

wifi_data_ac_256=scipy.io.loadmat('80211ac_5packets_256qam.mat')['waveStruct']['waveform'][0][0]
wifi_data_ac_256=wifi_data_ac_256.T
wifi_data_ac_256=wifi_data_ac_256.flatten()

# ************ loading 802.11ax file **************

wifi_data_ax_16=scipy.io.loadmat('80211ax_10packets_16qam.mat')['waveStruct']['waveform'][0][0]
wifi_data_ax_16=wifi_data_ax_16.T
wifi_data_ax_16=wifi_data_ax_16.flatten()

wifi_data_ax_64=scipy.io.loadmat('80211ax_10packets.mat')['waveStruct']['waveform'][0][0]
wifi_data_ax_64=wifi_data_ax_64.T
wifi_data_ax_64=wifi_data_ax_64.flatten()

wifi_data_ax_256=scipy.io.loadmat('80211ax_10packets_256qam.mat')['waveStruct']['waveform'][0][0]
wifi_data_ax_256=wifi_data_ax_256.T
wifi_data_ax_256=wifi_data_ax_256.flatten()


wifi_data_numpy=np.concatenate((wifi_data_ax_16,wifi_data_ax_64,wifi_data_ax_256,wifi_data_ac_16,wifi_data_ac_64,wifi_data_ac_256))


wifi_data=pd.DataFrame({'real':np.real(wifi_data_numpy),'img':np.imag(wifi_data_numpy),
                       'phase':np.angle(wifi_data_numpy), 'amp':np.abs(wifi_data_numpy),
                      })

# because some rows are 0, I'm removing them

wifi_data=wifi_data.loc[~(wifi_data==0).any(axis=1)]
wifi_data['label']=1
wifi_data.reset_index(drop=True, inplace=True)
wifi_data.head()



# loading recorded LTE data

lte_data_numpy=np.fromfile('usrp_lte.csv', dtype=np.complex64, count=500000)

lte_data_numpy=lte_data_numpy*100

lte_data=pd.DataFrame({'real':np.real(lte_data_numpy)[:],'img':np.imag(lte_data_numpy)[:],
                        'phase':np.angle(lte_data_numpy)[:], 'amp':np.abs(lte_data_numpy)[:],
                       })


# because some rows are 0, I'm removing them

lte_data=lte_data.loc[~(lte_data==0).any(axis=1)]
lte_data['label']=0
#lte_data=lte_data.sample(frac=1)
lte_data.reset_index(drop=True, inplace=True)
lte_data.head()


wifi_data.shape



# creating train and test data.

training_size = 350000

# train data. dropping labels on training data since we don't need it for training
x_train = lte_data[:training_size].drop(['label'], axis=1)

# test data
x_test=lte_data[training_size:]

# inserting Wi-Fi data in between
x_test.iloc[15000:25000] = wifi_data.iloc[:10000].values
x_test.iloc[25000:35000] = wifi_data.iloc[20000:30000].values
x_test.iloc[35000:45000] = wifi_data.iloc[40000:50000].values
x_test.iloc[70000:80000] = wifi_data.iloc[60000:70000].values
x_test.iloc[80000:90000] = wifi_data.iloc[70000:80000].values
x_test.iloc[90000:100000] = wifi_data.iloc[80000:90000].values


print("Length of train data:",len(x_train))
print("Length of test data:",len(x_test))


# creating training and validation dataset
x_train,x_validate = train_test_split(x_train,test_size=0.2,random_state=random_seed)

# separating labels from test dataset for plotting later
x_test,labels = x_test.drop('label',axis=1).values,x_test.label.values



# creating a pipeline for normalizing

pipeline=Pipeline([('normalizer',Normalizer()), ('scaler',MinMaxScaler())])

x_train_transformed=pipeline.fit_transform(x_train)
x_validate_transformed=pipeline.fit_transform(x_validate)


# plotting the relation between the 1st 3 features after transformation

# storing the column names because the transformed data is a numpy array and does not contain column name. But the  \
# iloc function needs a dataframe and so we are converting the numpy array to dataframe which needs the column name.
column_names=list(x_train.columns)

# one way of doing it
g=sns.PairGrid(pd.DataFrame(x_train_transformed, columns=column_names).iloc[:,:5].sample(600))   

# another way of doing the above without converting into dataframe. Note: .sample(600) is returning a random 600 examples not the first 600 examples
#g=sns.PairGrid(pd.DataFrame(x_train_transformed[:600,:3]))
plt.subplots_adjust(top=0.9)
g.fig.suptitle('After:')
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot)


#shape of training and testing data

print(x_train_transformed.shape)
print(x_validate_transformed.shape)



#developing model

input_dim = x_train_transformed.shape[1]


#encoder
input_layer=Input(shape=(input_dim,), name="input")
hidden_layer1=Dense(input_dim, activation='elu', name="hidden1")(input_layer)
hidden_layer2=Dense(3, activation='elu', name="hidden2")(hidden_layer1)

#code

code_layer1=Dense(2, activation='elu', name='code_layer1')(hidden_layer2)

#decoder

hidden_layer3=Dense(3, activation='elu', name="hidden3")(code_layer1)
hidden_layer4=Dense(input_dim, activation='elu', name="hidden4")(hidden_layer3)
output_layer=Dense(input_dim, activation='elu', name="output")(hidden_layer4)

auto_encoder = Model(input_layer,output_layer)
auto_encoder.compile(optimizer='adam', loss='mse', metrics=['acc'])
auto_encoder.summary()


# training the model

Epochs=15

start_time=time.time()

history = auto_encoder.fit(x_train_transformed,x_train_transformed, epochs = Epochs, batch_size=256, shuffle=True,
                       validation_data=(x_validate_transformed,x_validate_transformed))

end_time=time.time()



print("GPU time for training:",end_time-start_time,"s")



# plotting train and validation loss

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(Epochs)

plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# plotting train and validation accuracy

acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(Epochs)
plt.figure()
axes=plt.gca()
axes.set_ylim([0.5,1])
plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')

plt.title('Training and validation accuracy')
plt.legend()
plt.show()



# Reconstructing data

x_test_transformed=pipeline.transform(x_test)

reconstructions = auto_encoder.predict(x_test_transformed)

mse = np.mean(np.power((reconstructions-x_test_transformed),2),axis=1)


# calculating the modified z-score to find outliers

# manually setting threshold
Threshold= 0.008


def _z_score(mse):
    median_val_mse = np.median(mse)
    diff = np.abs(mse - median_val_mse)
    median_of_diff = np.median(diff)
    
    return 0.6745 * diff/median_of_diff


z_score = _z_score(mse)

outliers= mse > Threshold

print("There are {} outliers in a total of {} signals.".format(np.sum(outliers), len(z_score)))



fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(mse, label='MSE', color='b', linewidth=1)
plt.axhline(Threshold, label='Threshold', color='r', linestyle='-')
plt.legend(loc='upper left')
ax.set_title('Reconstruction loss graph in deep autoencoder', fontsize=16)
plt.xlabel('Test samples')
plt.ylabel('Loss')
plt.savefig('dense_autoencoder_singleMCS_multiprotocol.png')
plt.show


# Visualizing the latent space

encoder = Model(input_layer,code_layer1)

latent_space = encoder.predict(x_test_transformed)

X = latent_space[:,0]
Y = latent_space[:,1]

plt.subplots(figsize=(8,8))
plt.scatter(X[labels==0],Y[labels==0], s=5, c='g', alpha=0.5, label='LTE')
plt.scatter(X[labels==1],Y[labels==1], s=5, c='r', alpha=0.5, label='Wi-Fi')

plt.legend()
plt.title('Deep autoencoder\'s Latent Space Representation')

plt.savefig('Dense autoencoder\'s Latent Space Representation_singleMCS_multiprotocol.png')
plt.show()



# CONFUSION MATRIX

cm=confusion_matrix(labels,outliers)


ax=plt.subplot()
sns.set(font_scale=1.2)
sns.heatmap(cm,annot=True, fmt='g')

ax.set_xlabel('Predicted values')
ax.set_ylabel('True values')

ax.set_yticklabels(['LTE','Wi-Fi'])
ax.set_xticklabels(['LTE','Wi-Fi'])
ax.set_title('Confusion matrix for deep autoencoder')

plt.savefig('conf_matrix_dense_autoencoder_singleMCS_multiprotocol.png')


# Precision, Recall and F1-score

print("Precision of classification:", "%.2f" % (precision_score(labels,outliers)*100),"%")
print("Recall of classification:","%.2f" % (recall_score(labels,outliers)*100),"%")
print("F1-score of classification:","%.2f" % (f1_score(labels,outliers)))


