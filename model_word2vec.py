import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.models import Sequential
from CRF import CRF

class Model():
    """
    TODO: 增加字符向量+CNN步骤
    """
    def __init__(
        self, 
        batch_size=None,
        epochs=None,
    ):
        self.model = None
        self.batch_size=batch_size
        self.epochs=epochs

    def fit(self, X, Y, label_count):
        X = np.array(X)
        Y = np.array(Y)
        
        print(X.shape)
        print(Y.shape)
        
        # build model
        self.model = Sequential()
        self.model.add(Input(shape=(X.shape[1], X.shape[2])))
        # 防止过拟合
        self.model.add(Dropout(0.5))
        self.model.add(Bidirectional(LSTM(32, return_sequences=True)))
        # 防止过拟合
        self.model.add(Dropout(0.5))
        crf = CRF(label_count, name='crf_layer')
        self.model.add(crf)
        self.model.compile('adam', loss={'crf_layer': crf.get_loss}, metrics={'crf_layer': crf.get_accuracy})
        
        print(self.model.summary())
        
        X = tf.convert_to_tensor(np.array(X))
        Y = tf.convert_to_tensor(np.array(Y))
        self.model.fit(X, Y, batch_size=self.batch_size, epochs=self.epochs, validation_split=0.2)

    def predict(self, X):
        X = tf.convert_to_tensor(X)
        Y = self.model.predict(X)
        return Y

    def summary(self):
        return self.model.summary()
