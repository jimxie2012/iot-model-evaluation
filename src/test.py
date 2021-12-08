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
from jcnn import CJCNN
from common import preprocess,get_csv_files,g_analyse_path,g_ds_sample_path,g_model_path
from inference import get_all_models
from jcnn import get_features
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

os.environ['CUDA_VISIBLE_DEVICES']='1'

def main( ):
    
    csv_list = []
    get_csv_files(g_ds_sample_path,csv_list)
    csv_list.sort()
    all_models = get_all_models()
    for item in json.loads(all_models.to_json(orient="records")):
        model_name = item['model_name']
        if not model_name.strip().lower() == 'cnn':
            continue
        model_file = item['model_file']
        dataset = item['dataset']
        for csv in csv_list:
            df = pd.read_csv(csv,index_col=0)
            df_all = preprocess(df)
            df_all = df_all.sample(100)
            del df_all['label']
            data_set = os.path.splitext(os.path.basename(csv))[0]
            print("get feature ","%s(%s)"%(model_name,dataset), "data set %s"%data_set)

            features = get_features(model_file,df_all)
            for key in features:
                path = "/data/paper/features/%s-%s/"%(dataset,key)
                os.system("mkdir -p %s"%path)
                with open("%s%s.csv"%(path,data_set),'w') as fp:
                    fp.write(json.dumps(features[key].tolist()))

            break
        break

if __name__ == "__main__":
    main (  )
