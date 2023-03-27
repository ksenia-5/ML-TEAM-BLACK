# import pandas and keras modules
import pandas as pd
from tensorflow.keras import Sequential, Dense

mnist = pd.read_csv('mnist.csv')
# load in MNIST dataset (2500 of 60000 digit images)
# each one 28x28 pixels flattened to 784 array
# so data input shape is (784,)
# pixel data as X
# labels as y

# Create the model: model
model = Sequential()

# Add the first hidden layer
model.add(Dense(50, activation='relu',input_shape=(784,)))

# Add the second hidden layer
model.add(Dense(50, activation='relu'))

# Add the output layer
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(X, y, validation_split=0.3, epochs=10)