# Speech-Denoising-using-Deep-Learning
Speech Denoising Using Deep Learning

> The problem statement for the application is described as follows:
Usually, in communication systems, the received signal are polluted with noise and distortion, which are mainly caused due to channel behavior. Thus, at the receiving end, the signal may lose its information due to corruption. Thus, denoising of the received signal is essential for efficient communication in one-to-one as well as broadcast systems. 

Speech signal denoising, or noise reduction, is the removal of noise and distortions present in the signal for recovering back the original signal, free from noise. The objective is to improve the Signal-to-Noise Ratio (SNR) of the incoming speech audio. This is an important application of Digital Signal Processing, which has various uses such as in cellular phones, hearing aids, teleconferencing (has become widespread after COVID-19 pandemic) etc.

In this project, we worked using principles of Digital Signal Processing, namely, Short-Time Fourier Transform; along with exploring deep learning convolutional neural network by 1D CNN & 2D CNN 

We explored the principles and made comparisons to identify that 1D CNN outperforms 2D CNN.

> 1D CNN design implemented:

-> Two convolution layers with filters 16 and 32 respectively.
-> Also kernel sizes of 16, 8 respectively.
-> Same padding is used.
-> ReLU activation function is used in all the convolution layers.
-> Max pooling layers are implemented one each after the convolution layer.
-> Flattening is implemented for the last max pooling layer.
-> A dense layer of 513 units with a ReLU activation.
-> Adam optimizer, mean squared error loss function are used.
-> 1000 epochs are used for training.


> 2D CNN design implemented:

-> Input layer with TensorShape as (-1,20,513,1).
-> Two convolution layers used with filters of 16,32 are used respectively.
-> Also kernel size of (4,4) for all of the above layers.
-> ReLU activation is used in all the convolution layers.
-> Max pooling layers are used after each convolution layer, with pool_size of (2,2).
-> Final max pooling layer is flattened.
-> A dense layer is used with 513 hidden units with ReLU activation function.
-> Adam optimizer (0.0002 learning rate) and Mean squared error Loss function are used.
-> Batch size of 64 and 500 epochs for training.    

