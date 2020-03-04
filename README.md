# Neural Network Training
##### Training a Fully connected Feedforward Neural Network [aka MLP-Multi-Layer Perceptron] &amp; a Convolutional Neural Network [CNN]
> ###### The latter exploits local geometric aspects of the input values instead of connecting every unit to all those at the next level above.

##### By configuring each neural network, it will learn to classify various images of clothing [boot, sweater, etc.]. The dataset we can use is called Fashion MNIST which consists of greyscale [only 1 color channel] images of various articles of clothing. The images are 28x28 [28x28x1 when including the 1 color channel] in size and the features amongst some classes can be quite similar, making it difficult even for humans to get 100% accuracy.

<hr>

##### Will be using : Keras API [Official frontend of Tensorflow] &amp; Tensorflow [Google-supported backend]
> ###### Compatible w/ Python ```2.7 - 3.6```
##### Keras will have a Python implementation; To gain a better understanding of how to use Keras in your Python script, please refer to their [site](https://keras.io/)

<hr>

## Neural Network Examples for reference
#### Fully Connected NN
```python
import keras
from keras.models import Sequential
from keras.layers import Dense

# Convenient wrapper for creating a sequential model where the output of one layer feeds into the input of the next layer.
model = Sequential()

# ”Dense” here is the same as a fully connected network. Input dimension only needs to be specified for the first layer, may be a 
# tuple with more than one entries, e.g. (100, 100, 3) for an input consisting of color images (RedBlueGreen sub pixels, hence the 3) 
# that are 100x100 pixels in size. NOTE that the dataset we’ll be using has only 1 color channel, NOT 3 like in this example.
model.add(Dense(units=64, activation=’relu’, input_dim =100))
model.add(Dense(units=10, activation=’softmax’))

# Sets up the model for training and inferring.
model.compile(loss=’categorical_crossentropy’, optimizer=’sgd’, metrics=[’accuracy’])

# The training is done here; could be broken up explicitly into branches.
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Get loss (value of loss function; lower is better) and any metrics specified in the compile step, such as accuracy. The test
# batch consists of held out data that you want to verify your network on. You should NEVER use test data in the training period.
# It violates standards, ethics and the credibility of your results.
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

# Get class outputs for the test batch size of 128.
classes = model.predict(x_test, batch_size=128)
```

#### Convolutional NN
```python
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

model = Sequential()

# input: 28x28 images with 1 color channel -> (28, 28, 1) tensors.
# This applies 32 convolutional filters of size 3x3 each with 'relu' activation after the convolutionals are done.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e − 6, momentum=0.9 , nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

# x_train and y_train not shown above . . . these are your inputs and outputs for training.
model.fit(x_train, y_train, batch_size=32, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=32)
```

<hr>

##### Main file to run from the command line to train and validate your models : `train_test_proj1.py`
###### Example of flags it will accept : `train_test_proj1.py   -b 32   -e 10   -n conv   -o adam`
> ###### The -b flag is for the batch size the network uses
> ###### The -e flag is for the number of epochs the network will run
> ###### The -n flag determines what type of network to make : 'mlp' - fully connected and 'conv' - convolutional
> ###### The -o flag is for what type of optimizer to use : 'sgd' - stochastic gradient descent, 'adam' - adam, and 'rmsprop' - rmsprop
> ###### The -s flag [Not shown above] w.o anything following allows for a snapshot of the weights of a network to be loaded back into the network.
> ###### The -t flag [Not shown above] w/o anything following can be used to just test the model you're loading in with the -s flag; an accuracy score on the test set will be printed out.

##### After training, the total time for all training [which includes past episodes of training] will be output.

<hr>

#### You can now *edit* the implemented fully connected &amp; convolutional neural networks in the file : [`training_model.py`](https://github.com/HG7777/NeuralNetworkTraining/blob/master/training_model.py)

#### Brief Analysis of current implementation can be found in the file : [`model_analysis.pdf`](https://github.com/HG7777/NeuralNetworkTraining/blob/master/model_analysis.pdf)
