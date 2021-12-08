import tensorflow as tf
import os,sys,json
from common import get_csv_files,preprocess,is_sample_qualified,g_model_path,g_ds_sample_path
import pickle as pk
import numpy as np
import pandas as pd
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,AveragePooling2D,Flatten,Dense
from tensorflow.keras.layers import Conv1D, MaxPool1D,Dropout,Activation
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

def getPrecision(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))#TP
    N = (-1)*K.sum(K.round(K.clip(y_true-K.ones_like(y_true), -1, 0)))#N
    TN=K.sum(K.round(K.clip((y_true-K.ones_like(y_true))*(y_pred-K.ones_like(y_pred)), 0, 1)))#TN
    FP=N-TN
    precision = TP / (TP + FP + K.epsilon())#TT/P
    return precision
def getRecall(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))#TP
    P=K.sum(K.round(K.clip(y_true, 0, 1)))
    FN = P-TP #FN=P-TP
    recall = TP / (TP + FN + K.epsilon())#TP/(TP+FN)
    return recall

class CJCNN(object):

    def __init__(self):
        self.Create()

    def Create(self,n_features = 37):
        self.m_model = Sequential()
        self.m_model.add(Conv1D(1024, 3, activation='relu', input_shape=(37,1), padding="same"))
        #self.m_model.add(MaxPool1D(pool_size=3, strides=3))
        self.m_model.add(Conv1D(512, 3, strides=1, activation='relu', padding='same'))
        #self.m_model.add(MaxPool1D(pool_size=3, strides=3))
        self.m_model.add(Conv1D(256, 3, strides=1, activation='relu', padding='same'))
        self.m_model.add(Conv1D(128, 3, strides=1, activation='relu', padding='same'))
        self.m_model.add(Conv1D(64, 3, strides=1, activation='relu', padding='same'))

        #self.m_model.add(Dropout(0.25))
        self.m_model.add(Flatten())
        self.m_model.add(Dense(1024))
        self.m_model.add(Dense(120, activation='tanh'))
        self.m_model.add(Dense(84, activation='tanh'))
        self.m_model.add(Dense(2, activation='softmax'))
        return self.m_model

    def Summary(self):
        return self.m_model.summary()

    def Train(self,df_train,df_valid):
        df_1 = df_train.copy( deep = True )
        df_2 = df_valid.copy( deep = True )
        train_labels = df_1['label'].to_numpy()
        del df_1['label']
        train_input = df_1.to_numpy()
        train_input = train_input.reshape(-1,37,1)
        train_labels = train_labels.reshape(-1,1)

        test_labels = df_2['label'].to_numpy()
        del df_2['label']
        test_input = df_2.to_numpy()
        test_input = test_input.reshape(-1,37,1)
        test_labels = test_labels.reshape(-1,1)

        sgd = tf.keras.optimizers.SGD(lr=0.1, decay=1e-4, momentum=0.99, nesterov=True)
        self.m_model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        print("train input shape",train_input.shape)
        measure = self.m_model.fit(train_input, \
                                   train_labels, \
                                   validation_data = (test_input,test_labels),\
                                   batch_size=100,\
                                   epochs=20,\
                                   verbose=1)
        score = self.m_model.evaluate(test_input, test_labels, verbose=1)
        print("model score:", score)

    def Save(self,model_file):
        self.m_model.save(model_file)

    def Load(self,model_file):
        self.m_model = load_model(model_file)

    def GetLayerCount(self):
        return len(self.m_model.layers)

    def GetLayerCount(self):
        return len(self.m_model.layers)

    def Predict(self,df_test,layer_number = -1):
        df_valid = df_test.copy( deep = True )
        #test_labels = df_valid['label'].to_numpy()
        #del df_valid['label']
        test_input = df_valid.to_numpy()
        test_input = test_input.reshape(-1,37,1)
        #test_labels = test_labels.reshape(-1,1)

        layer_model = Model(inputs=self.m_model.input, outputs=self.m_model.layers[layer_number].output)

        p = layer_model.predict(test_input)
        y_pred = np.argmax( p , axis=1)
        return y_pred

def train_cnn(sample_name):
    csv_file = "%s/%s.csv"%(g_ds_sample_path , sample_name)
    cnn_root = "%s%s/cnn/"%(g_model_path,sample_name)
    model_file = "%ssklearn_model_python_cnn.h5"%(cnn_root)
    
    os.system("mkdir -p %s"%(cnn_root))
    print("cnn train begin" , sample_name)
    print(csv_file,cnn_root,model_file)
    df = pd.read_csv(csv_file , index_col=0)
    df_all = preprocess(df)

    df_train = df_all.sample(frac=0.85)
    df_valid = df_all.drop(df_train.index)

    cnn = CJCNN()
    cnn.Create()
    cnn.Train( df_train , df_valid )
    cnn.Save( model_file )
    print("cnn train end" , sample_name)

def test():
    train_sample = '/data/paper/model/s-0-1-0/train.csv'
    valid_sample = '/data/paper/model/s-0-1-0/valid.csv'
    test_sample = '/data/paper/model/s-0-2-0/valid.csv'
    df_train = pd.read_csv(train_sample,index_col=0)
    df_valid = pd.read_csv(valid_sample,index_col=0)
    df_test = pd.read_csv(test_sample,index_col=0)
    cnn = CJCNN()
    cnn.Create()
    print(cnn.Summary())
    cnn.Train(df_train,df_valid)
    cnn.Save("./lenet-5-model.h5")
    cnn.Load("./lenet-5-model.h5")

    print("layer count ",cnn.GetLayerCount())
    ret = cnn.Predict( df_test , 7 )
    print(ret)

def train():
    os.system("mkdir -p %s"%g_model_path)
    csv_lsit = []
    get_csv_files( g_ds_sample_path , csv_lsit )
    csv_lsit.sort()
    for fi in csv_lsit:
        sample_name = os.path.splitext(os.path.basename(fi))[0]
        if sample_name.split("-")[-1] == '0':
            print("begin train ",sample_name)

            train_cnn( sample_name )

            print("end train ",sample_name)

def get_features( mode_file, df ):
    cnn = CJCNN()
    cnn.Load( mode_file )
    features = {}
    for i in [0,1,2,3,4,6,7,8,9]:
        features["F-%d"%i] = cnn.Predict( df , i )
    return features

def main():

    get_features()

if __name__ == "__main__":
    main()
