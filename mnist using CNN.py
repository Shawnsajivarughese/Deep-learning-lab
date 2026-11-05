
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize images
x_train = x_train.reshape(-1,28,28,1) / 255.0
x_test = x_test.reshape(-1,28,28,1) / 255.0

# One hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# MODEL 1 - One convolution layer
model1 = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128,activation='relu'),
    Dense(10,activation='softmax')
])

# Compile and train Model 1
model1.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model1.fit(x_train,y_train,epochs=5,batch_size=64)

# Evaluate Model 1
loss1, acc1 = model1.evaluate(x_test,y_test)
pred1 = model1.predict(x_test[:2])

print("MODEL 1 RESULTS")
print("Accuracy:",acc1*100)
print("Loss:",loss1)

# Figure 1 - Model 1 predictions
plt.figure(figsize=(8,4))

for i in range(2):
    plt.subplot(1,2,i+1)
    plt.imshow(x_test[i],cmap='gray')
    plt.title("Predicted: " + str(pred1[i].argmax()))
    plt.axis('off')

plt.show()


# MODEL 2 - Two convolution layers
model2 = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128,activation='relu'),
    Dense(10,activation='softmax')
])

# Compile and train Model 2
model2.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model2.fit(x_train,y_train,epochs=5,batch_size=64)

# Evaluate Model 2
loss2, acc2 = model2.evaluate(x_test,y_test)
pred2 = model2.predict(x_test[:2])

print("MODEL 2 RESULTS")
print("Accuracy:",acc2*100)
print("Loss:",loss2)

# Figure 2 - Model 2 predictions
plt.figure(figsize=(8,4))

for i in range(2):
    plt.subplot(1,2,i+1)
    plt.imshow(x_test[i],cmap='gray')
    plt.title("Predicted: " + str(pred2[i].argmax()))
    plt.axis('off')

plt.show()
