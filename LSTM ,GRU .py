import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,SimpleRNN,LSTM,GRU,Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=10000)

x_train = pad_sequences(x_train,maxlen=200)
x_test = pad_sequences(x_test,maxlen=200)

result = []
best_accuracy = 0
lowest_loss = float('inf')


for name,layer in (("Simple RNN",SimpleRNN),("LSTM",LSTM),("GRU",GRU),):
    model = Sequential([Embedding(10000,64),layer(64),Dense(1,activation='sigmoid')])
    model.compile(optimizer = 'adam', loss='binary_crossentropy' , metrics = ['accuracy'])

    print("\ntraining:",name)

    model.fit(x_train,y_train,epochs=3,batch_size=64)
    loss,accuracy = model.evaluate(x_test,y_test)
    
    print('test accuracy:',accuracy *100)
    print('test loss:',loss *100)
    result.append([name,loss,accuracy])

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_acc_model = name

    if loss < lowest_loss:
        lowest_loss = loss
        best_loss_model = name

print("\nfinal analysis")
print('best_accuracy',best_acc_model,best_accuracy *100)
print('lowest_loss',best_loss_model,lowest_loss *100)
