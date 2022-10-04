# Speech Denoising using Deep Learning

import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#os.chdir('D:\IUB\Deep Learning\Assignments\Assignment-1\data')

#Loading training and testing files
#Computing STFT on all the files
s, sr = librosa.load('train_clean_male.wav', sr=None)
S = librosa.stft(s, n_fft=1024, hop_length=512)

sn, sr = librosa.load('train_dirty_male.wav', sr=None)
X = librosa.stft(sn, n_fft=1024, hop_length=512)

x_test, sr = librosa.load('test_x_01.wav', sr=None)
X_test = librosa.stft(x_test, n_fft=1024, hop_length=512)

x_test2, sr = librosa.load('test_x_02.wav', sr=None)
X_test2 = librosa.stft(x_test2, n_fft=1024, hop_length=512)

#Calculating the magnitude of all the input files
mag_S = np.abs(S)
mag_X = np.abs(X)
mag_X_test = np.abs(X_test)
mag_X_test2 = np.abs(X_test2)

#Defining model specifications
learning_rate = 0.001
act_layers = [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu]
neurons = [513, 513, 513, 513]
num_layers = len(act_layers)

#Generating a deep network of n layers with specific activation functions
#and specified number of neurons in each layer
def getModel(x , act_layers , neurons):
    num_layers = len(act_layers)
    layers = [0]*num_layers
    
    for i in range(0 , len(act_layers)):        
        if i == 0:
            layers[i] = tf.layers.dense(x , units= neurons[i] , activation=act_layers[i])        
        elif i < num_layers-1:
            layers[i] = tf.layers.dense(layers[i-1] , units= neurons[i] , activation=act_layers[i])
        else:
            layers[i] = tf.layers.dense(layers[i-1] , units= neurons[i] , activation=act_layers[i])
    
    return layers

#Creating placeholders for input and output
input = tf.placeholder(tf.float32, [None, 513])
labels = tf.placeholder(tf.float32, [None, 513])

output = getModel(input, act_layers, neurons)

#Defining the loss function along with its optimizer
loss = tf.reduce_mean(tf.square(output[num_layers - 1]-labels))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

count = 0
batch_size = 100
flag = True

while flag:
    size = 0
    #Mini batching with the given batch size
    for i in range(0 , 2500, batch_size):
        size += batch_size
        if size <= 2459:
            batch_x = mag_X[:,i : size]
            batch_y = mag_S[:,i : size]
        else:
            batch_x = mag_X[:,i : 2459]
            batch_y = mag_S[:,i : 2459]
       
        
        feed_dict = {input: batch_x.T, labels: batch_y.T}
        train_step.run(feed_dict=feed_dict)
   
    if count%10 == 0:             
        loss_calc = loss.eval(feed_dict=feed_dict)
        print("Epoch %d, loss %g"%(count, loss_calc))
    
    #Once 100 epochs are completed, training is stopped
    if count >= 100:
        flag = False  
        
    count+=1

#Calculating the output from the given input, trained model and layer number
def feedforward(input_data, dnn_output , layer_num):
    output = dnn_output[layer_num - 1].eval(feed_dict = {input : input_data})
    
    return output

#Recovering the complex values of the file from the output of the model
def recover_sound(X , mag_X , mag_output):
  temp = X / mag_X
  s_hat = temp * mag_output
  
  return s_hat

#Computing the output from the model for both the test files
s_hat_test1 = feedforward(mag_X_test.T , output , 4)
s_hat_test2 = feedforward(mag_X_test2.T , output , 4)

#Recovering the complex values of both the test files
s_hat1 = recover_sound(X_test , mag_X_test , s_hat_test1.T)
s_hat2 = recover_sound(X_test2 , mag_X_test2 , s_hat_test2.T)

#Reconstructing the test files after removing noise
recon_sound = librosa.istft(s_hat1 , hop_length=512 , win_length=1024)
librosa.output.write_wav('test_s_01_recons.wav', recon_sound, sr)

recon_sound2 = librosa.istft(s_hat2 , hop_length=512 , win_length=1024)
librosa.output.write_wav('test_s_02_recons.wav', recon_sound2, sr)

#For testing purpose, feeding the model with train_dirty_male file
#From the output generated, reconstructing the audio file
s_hat_test3 = feedforward(mag_X.T , output , 4)
s_hat3 = recover_sound(X, mag_X , s_hat_test3.T)
recon_sound3 = librosa.istft(s_hat3 , hop_length=512 , win_length=1024)
size_recon_sound3 = np.shape(recon_sound3)[0]

#Once the audio file is generated, calculating the SNR value
s = s[: size_recon_sound3]
num = np.dot(s.T , s)
den = np.dot((s - recon_sound3).T,(s - recon_sound3))
SNR = 10 * np.log10(num/den)
print('Value of SNR : ' + str(SNR))
