"""
    @Sathish-V

    Beginner-level ANN(Artificial Neural Network) Project

    Description:

        The project helps beginners to understand how ANN works , here I've taken a simple Regression based
    use case which is 2nd table, there are 2 hidden-layers each with 10-units and the weights are
    initialized by the he_uniform distribution. Activation function used here is ReLu.

"""

#import necessary modules

import numpy as np
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


#preparing dataset which is table for number '2'

x = np.array([[x, 2] for x in range(1, 100001)])
y = np.array([x * 2 for x in range(1, 100001)])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=32)


#Building ANN model

model = keras.Sequential()

model.add(layers.Dense(units=10, kernel_initializer="he_uniform", activation="relu", input_dim=2))

model.add(layers.Dense(units=10, kernel_initializer="he_uniform", activation="relu"))

model.add(layers.Dense(units=1, activation="linear"))

model.compile(optimizer="adam", loss="mean_absolute_error", metrics=["mean_absolute_error"])

#Fit the model

model_details = model.fit(x_train, y_train, validation_split=0.15, epochs=100)

#prediction

prediction = model.predict(x_test)

#Calculating the metrics

pred = np.array([x[0] for x in prediction])

mae = mean_absolute_error(pred, y_test)
print("Mean Absolute Error:", mae)


#Plotting the actual and predicted values corresponding to the independent variable

input = np.array([[x , 2] for x in range(1, 21)])

ouput = model.predict(input)

y_axis = [x[0] for x in ouput]

plt.subplot(1, 2, 1)
plt.scatter(input[:, 0], input[:, 0] * 2)
plt.title("Actual-value")
plt.xlabel("2 x i")
plt.ylabel("product of 2")

plt.subplot(1, 2, 2)
plt.scatter(input[:, 0], ouput)
plt.title("Predicted-value")
plt.xlabel("2 x i")
plt.ylabel("product of 2")

plt.show()