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
```
