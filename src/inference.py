# -*- coding: utf-8 -*-
"""
    This script is used to inference all sample data

    Edit by Jim Xie (xiewenwei@sina.com)  2021/11/28
"""
import h2o
import time,json,os,sys
import numpy as np
import pandas as pd
from jsvm import CJSVM
from common import preprocess,get_csv_files,g_analyse_path,g_ds_sample_path,g_model_path
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import fbeta_score
from warnings import filterwarnings
filterwarnings("ignore") 
np.set_printoptions(suppress=True)
pd.set_option('display.float_format',lambda x : '%.8f' % x)

#scan path and find all trinned models files
def get_model_files(filepath,file_list=[]):
    files = os.listdir(filepath)
    for fi in files:
        fi_d = os.path.join(filepath,fi)
        if os.path.isdir(fi_d):
            get_model_files(fi_d,file_list)
        else:
            if fi_d.find("_model_python_") > 0:
                file_list.append(fi_d)
                
#get all trained model information
def get_all_models():
    model_list = []
    model_file_list = []
    model_path = g_model_path
    get_model_files(model_path , model_file_list)
    for fi in model_file_list:
        t = fi.split("/")
        model_file = fi
        model_name = t[-2]
        dataset =  t[-3]
        train_set = "%s/%s/train.csv"%(model_path,dataset)
        valid_set = "%s/%s/valid.csv"%(model_path,dataset)
        temp = {}
        temp['model_file'] = model_file
        temp['model_name'] = model_name
        temp['dataset'] = dataset
        temp['train_set'] = train_set
        temp['valid_set'] = valid_set
        model_list.append(temp)
    return pd.DataFrame(model_list)

class CJPerformance(object):
    
    def __init__(self,model_file):
        self.m_model_file = model_file
        if not self.is_svm():
            h2o.init(ip="localhost" , port=54321 , min_mem_size = "24G")
            self.m_model = h2o.load_model(model_file)
        else:
            self.m_model = CJSVM()
            self.m_model.load(model_file)
    
    def is_svm(self):
        if self.m_model_file.find("sklearn_model_python_svm.pkl") >= 0:
            return True
        else:
            return False
    
    def predict(self , sample_file , need_preprocess = True ):
        df = pd.read_csv(sample_file,index_col=0)
        if need_preprocess:
            df_all = preprocess(df)
        else:
            df_all = df.copy( deep = True )
        if not self.is_svm():
            df_h2o = h2o.H2OFrame(df_all)
            x = df_h2o.columns
            y = "label"
            #x.remove(y)
            df_h2o[y] = df_h2o[y].asfactor()
            pred = self.m_model.predict(df_h2o).as_data_frame()
            y_true = df_all[y].to_list()
            y_pred = pred['predict']
        else:
            X = df_all.copy( deep = True )
            Y = X['label']
            del X['label']
            y_true = df_all['label'].to_list()
            y_pred = self.m_model.predict(X)
            
        return y_true,y_pred
    
    def Test(self , sample_file , need_preprocess = True):
        y_true , y_pred = self.predict( sample_file , need_preprocess )
        result = {}
        result['confusion_matrix'] = confusion_matrix(y_true,y_pred)
        result['roc_curve'] = roc_curve(y_true,y_pred)
        result['recall'] = recall_score(y_true,y_pred)
        result['mcc'] = matthews_corrcoef(y_true,y_pred)
        result['accuracy'] = accuracy_score(y_true,y_pred)
        result['precision'] = precision_score(y_true,y_pred)
        result['auc'] = roc_auc_score(y_true,y_pred)
        result['log_loss'] = log_loss(y_true,y_pred)
        result['f1_score'] = f1_score(y_true,y_pred)
        result['fbeta_score'] = fbeta_score(y_true,y_pred,beta=0.5)
        return pd.Series(result)


def inference( item , csv_list):
    all_result = []
    model_name = item['model_name']
    dataset = item['dataset']
    model_file = item['model_file']
    print(model_name,dataset)
    perf = CJPerformance(model_file)
    for csv in csv_list:
        valid_set = os.path.splitext(os.path.basename(csv))[0]
        print("inference ","%s(%s)"%(model_name,dataset), "validation set %s"%valid_set)
        result = perf.Test(csv,True)
        result['model'] = model_name
        result['train'] = dataset
        result['valid'] = valid_set
        print(result)
        all_result.append(result)
    return all_result

def main( trained_model_name ):
    os.system("mkdir -p %s"%g_analyse_path)
    
    csv_list = []
    get_csv_files(g_ds_sample_path,csv_list)
    csv_list.sort()
    all_models = get_all_models()
    all_result = []
    for item in json.loads(all_models.to_json(orient="records")):
        model_name = item['model_name']
        if model_name.strip().lower() == trained_model_name:
            tmp_result = inference( item , csv_list  )
            all_result.extend(tmp_result)
            break
    df_result = pd.DataFrame(all_result)
    df_result.to_csv("%s%s.csv"%(g_analyse_path,trained_model_name))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("please run with model name , for example: python inference.py svm")
    else:
        model_name = sys.argv[1].strip().lower()
        main ( model_name )
