# -*- coding: utf-8 -*-
"""
    This script is used to train/inference svm models:

    Edit by Jim Xie (xiewenwei@sina.com)  2021/11/28
"""
import time,json,os
import numpy as np
import pandas as pd
from common import get_csv_files,preprocess,is_sample_qualified,g_model_path,g_ds_sample_path
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

class CJSVM(object):
    
    def __init__(self):
        n_estimators = 500
        self.m_clf = BaggingClassifier(SVC(probability=True,verbose=False),\
                                       max_samples=1.0 / n_estimators, \
                                       max_features=1.0, \
                                       n_estimators=n_estimators, \
                                       verbose=True,\
                                       n_jobs = 22)
    def train(self,df):
        start = time.time()
        X = df.copy( deep = True )
        Y = X['label']
        del X['label']
        print(Y.value_counts())
        self.m_clf.fit(X, Y)
        end = time.time()
        print("Finished Bagging SVC", end - start, self.m_clf.score(X,Y))
    
    def save(self,model_file):
        joblib.dump(self.m_clf, model_file)
    
    def load(self,model_file):
        self.m_clf = joblib.load(model_file)
        
    def predict(self,x):
        return self.m_clf.predict(x)

def train_svm(sample_name):

    csv_file = "%s/%s.csv"%(g_ds_sample_path , sample_name)
    svm_root = "%s%s/svm/"%(g_model_path,sample_name)
    model_file = "%ssklearn_model_python_svm.pkl"%(svm_root)

    os.system("mkdir -p %s"%(svm_root))

    print("svm train begin" , sample_name)
    print(csv_file)
    print(svm_root)
    print(model_file)

    df = pd.read_csv(csv_file , index_col=0)
    df_all = preprocess(df)
    svm = CJSVM()
    svm.train(df_all)
    svm.save( model_file )
    print("svm train end" , sample_name)

def test():
    test_file = "/data/dataset/sample/base-0-1-train-1.csv"
    model_file = "/data/dataset/model/base-1-1-train-1/customized/svm/sklearn_model_python_svm.pkl"
    df = pd.read_csv(test_file,index_col=0)
    df_all = preprocess(df)

    y_true = df_all['label']
    del df_all['label']

    test = CJSVM()
    test.load(model_file)
    y_pred = test.predict(df_all)
   
    print(confusion_matrix(y_true,y_pred))
    print(recall_score(y_true,y_pred))
    
def main():
    os.system("mkdir -p %s"%g_model_path)
    csv_lsit = []
    get_csv_files( g_ds_sample_path , csv_lsit )
    csv_lsit.sort()
    for fi in csv_lsit:
        sample_name = os.path.splitext(os.path.basename(fi))[0]
        if sample_name.split("-")[-1] == '0':
            print("begin train ",sample_name)
            train_svm(sample_name)
            print("end train ",sample_name)

if __name__ == "__main__":
    main()
