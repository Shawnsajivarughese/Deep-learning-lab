import pandas as pd
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

# Load dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Convert labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

results = []
# Experiments
experiments = [
    ("Baseline", "glorot_uniform", False, None),
    ("Xavier", "glorot_uniform", False, None),
    ("Kaiming", "he_normal", False, None),
    ("Dropout", "glorot_uniform", True, None),
    ("L2", "glorot_uniform", False, l2(0.001))
]

for name, init, drop, reg in experiments:

    for opt in ["adam", "sgd"]:

        model = Sequential([
            Flatten(input_shape=(32, 32, 3)),
            Dense(512, activation="relu", kernel_initializer=init, kernel_regularizer=reg),
            Dense(256, activation="relu", kernel_regularizer=reg),
            Dense(64, activation="relu", kernel_regularizer=reg),
            Dense(10, activation="softmax")
        ])
        

        if drop:
            model.add(Dropout(0.5))

        model.compile(optimizer=opt,
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])

        model.fit(x_train, y_train, epochs=5, batch_size=64)

        loss, acc = model.evaluate(x_test, y_test, verbose=0)

        results.append([name, opt, loss, acc * 100])

# Display results
df = pd.DataFrame(results, columns=[
    "Model", "Optimizer", "Loss", "Accuracy"
])
print(df)
