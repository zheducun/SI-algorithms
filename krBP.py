import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adam


class BPNN:
    
    def __init__(self, inodes=1, onodes=1, bs=100):
        self.inodes = inodes
        self.onodes = onodes
        self.bs = bs
        self.NN()
        
    def NN(self):
        X = Input(shape=(self.inodes, ), name="Linput")
        x = Dense(units=128, activation="relu", name="Lhidden1")(X)
        x = Dense(units=256, activation="relu", name="Lhidden2")(x)
        x = Dense(units=512, activation="relu", name="Lhidden3")(x)
        y = Dense(units=self.onodes, activation="relu", name="Loutput")(x)
        self.model = Model(inputs=X, outputs=y)

    def train(self, x_train, y_train, x_test, y_test, epochs=100):
        self.model.compile(loss="mse", optimizer=Adam(0.001))
        model_check = ModelCheckpoint('./model.h5', monitor='val_mae',
                                      verbose=0, save_best_only=True,
                                      save_weights_only=False,
                                      mode='auto', save_freq=1)
        h = self.model.fit(x=x_train, y=y_train,
                           validation_data=(x_test, y_test),
                           batch_size=self.bs, epochs=epochs, verbose=2)
        return h
