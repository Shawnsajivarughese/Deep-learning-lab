from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
vocal_size =10000
max_length=200
(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=max_length)

x_train=pad_sequences(x_train,maxlen=max_length)
x_test=pad_sequences(x_test,maxlen=max_length)
model=Sequential([Embedding(vocal_size,32,input_length=vocal_size),
                  SimpleRNN(32),Dense(1,activation='sigmoid')])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=5,batch_size=64)
loss,accuracy=model.evaluate(x_test,y_test,batch_size=32)
print("loss",loss)
print("accuracy",accuracy)
pred=model.predict(x_test[:1])[0][0]
print("positive" if pred>=0.5 else "negative",pred)
