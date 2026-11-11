import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Load dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Convert labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Class names
class_names = ["Airplane", "Automobile", "Bird", "Cat", "Deer",
               "Dog", "Frog", "Horse", "Ship", "Truck"]

# Hidden layers and activation functions
hidden_units = [(1024,512,256), (512,256,128),
                (256,128,64), (128,64,32)]

activations = ["relu", "tanh", "sigmoid"]

best_acc = 0
results = []

# Train models
for units in hidden_units:
    for act in activations:

        model = Sequential([
            Input(shape=(32,32,3)),
            Flatten(),
            Dense(units[0], activation=act),
            Dense(units[1], activation=act),
            Dense(units[2], activation=act),
            Dense(10, activation="softmax")
        ])

        model.compile(optimizer="adam",
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])

        model.fit(x_train, y_train, epochs=5,
                  batch_size=64, verbose=0)

        loss, acc = model.evaluate(x_test, y_test, verbose=0)

        results.append([units, act, loss, acc*100])

        if acc > best_acc:
            best_acc = acc
            best_model = model

# Display results
print("MODEL COMPARISON")

for r in results:
    print(r)

print("BEST ACCURACY:", best_acc * 100)

# Predict images
pred = best_model.predict(x_test[:5], verbose=0)

# Basic plot
for i in range(5):
    plt.imshow(x_test[i])
    plt.title("Pred: " + class_names[np.argmax(pred[i])] +
              "\nActual: " + class_names[np.argmax(y_test[i])])
    plt.axis("off")
    plt.axis("off")
    plt.show()
