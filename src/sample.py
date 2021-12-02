# -*- coding: utf-8 -*-
"""
    This script is used to generate tain/test samples from ToN and Apo123 dataset, including 3 steps:

    1. Create base dataset by sample with frac.
       s-0: more normal samples. ( full sample from ToN )
       s-1: more abnormal samples ( weight sample from apo123 )
       s-2: balance samples ( weighted and replaced sample from apo123 )
       s-3: more normal from apo 123 (weighted and replaced sample from apo123)
 
    2. Create train/test samples from base dataset (base-0,base-1,base-2,base-3)
       base-0-*: diffrent dataset, 500000 sampling from base-0 with diffrent frac
       base-1-*: more abnormal, 500000 sampling from base-0 with diffrent frac
       base-2-*: balance sample, 500000 sampling from base-0 with diffrent frac
       base-3-*: more normal, 500000 sampling from base-0 with diffrent frac

    3. Change sample format to one-hot coding
       create one-hot code map for 'conn_state','service','proto'
       change fields ('conn_state','service','proto') to one-hot

    Edit by Jim Xie (xiewenwei@sina.com)  2021/11/28
"""

import time,json,os
import numpy as np
import pandas as pd
from common import create_one_hot_map,create_one_hot_column,get_csv_files
from common import g_ds_base_path,g_ds_sample_path,g_ds_ton_common_csv_path,g_ds_apo_common_csv_path

def get_file_type(input_file):
    return os.path.splitext(os.path.basename(input_file))[0]

def get_label_count(df):
    df_test = df.groupby('label')['label'].count()
    json_count = df_test.reset_index(name='counts').set_index('label').to_json()
    json_count = json.loads(json_count)
    if not 'benign' in json_count['counts']:
        json_count['counts']['benign'] = 0
    if not 'abnormal' in json_count['counts']:
        json_count['counts']['abnormal'] = 0
    return json_count['counts']['benign'],json_count['counts']['abnormal']

def get_abnormal_rate(df):
    df_test = df.groupby('label')['label'].count()
    json_count = df_test.reset_index(name='counts').set_index('label').to_json()
    json_count = json.loads(json_count)
    if not 'benign' in json_count['counts']:
        json_count['counts']['benign'] = 0
    all_count = df.shape[0]
    normal_count = json_count['counts']['benign']
    normal_rate = normal_count/all_count
    abnormal_rate = (all_count - normal_count)/all_count
    return normal_rate,abnormal_rate

def get_sample(df_all,frac_or_n,normal_rate,abnormal_rate,is_replace):
    df_tmp = df_all.copy(deep=True)
    df_tmp['freq'] = df_tmp['label']
    mask = (df_tmp['label']=='Benign')|(df_tmp['label']=='benign')
    df_tmp.loc[mask, 'freq'] = normal_rate
    mask = (df_tmp['label']!='Benign')&(df_tmp['label']!='benign')
    df_tmp.loc[mask, 'freq'] = abnormal_rate
    print("get sample ",frac_or_n,normal_rate,abnormal_rate)
    if frac_or_n <= 1:
        return df_all.sample(frac=frac_or_n,weights=df_tmp['freq'].values,replace = is_replace)
    else:
        return df_all.sample(n=frac_or_n,weights=df_tmp['freq'].values, replace = is_replace)

def filter_dataset(df , kind , rate = 0.2):
    #normal sampling
    if kind == 0 :
        if df.shape[0] >= 10000:
            return df.sample(frac = rate )
        else:
            return df

    # more abnormal ( weighted sampling)
    if kind == 1:
        normal_rate,abnormal_rate = get_abnormal_rate(df)
        if abnormal_rate == 0:
            normal_rate = 0
            abnormal_rate = 1
        return get_sample(df,rate,abnormal_rate,normal_rate,False)

    #banlance sampling ( weighted sampling + replace sampling )
    if kind == 2:
        normal_rate,abnormal_rate = get_abnormal_rate(df)
        if abnormal_rate == 0:
            normal_rate = 0
            abnormal_rate = 1
        return get_sample(df,rate,abnormal_rate,normal_rate,True)

    #more normal ( weighted sampling + replace sampleing )
    if kind == 3:
        normal_rate,abnormal_rate = get_abnormal_rate(df)
        if abnormal_rate == 0:
            normal_rate = 0
            abnormal_rate = 1
        return get_sample(df,rate,2 * abnormal_rate,normal_rate,True)

def create_dataset( kind , rate= 0.2 ):
    csv_files =[]
    if kind == 0:
        get_csv_files( g_ds_ton_common_csv_path , csv_files )
    else:
        get_csv_files( g_ds_apo_common_csv_path , csv_files )
    print(csv_files)
    df_all = pd.DataFrame()
    for fi in csv_files:
        file_type = get_file_type(fi)
        df_tmp = pd.read_csv(fi,index_col=0)
        print(file_type,df_tmp.shape)
        if kind == 0 :
            df_tmp = df_tmp.sample(500000,replace=True)
        df_tmp = filter_dataset(df_tmp,kind,rate)
        df_all = pd.concat([df_all,df_tmp], ignore_index = True )
    df_all.reset_index(drop = True)
    df_all.to_csv("%s/s-%d.csv"%(g_ds_base_path,kind))

def create_sample(base_file, rate , n = 500000, scenario = 1):
    df = pd.read_csv(base_file,index_col = 0 ).sample( n )
    normal_count,abnormal_count = get_label_count(df)
    print(base_file,df.shape,normal_count,abnormal_count,"normal:abnormal ",normal_count/abnormal_count)
    df1 = pd.DataFrame()

    #scenario 1: diffrent dataset
    if scenario == 1:  
        df1 = get_sample(df , rate , normal_count , abnormal_count , False)

    #scenario 2: more abnormal
    if scenario == 2: 
        df1 = get_sample(df , rate , normal_count , abnormal_count , False)

    #scenario 3: balance sample
    if scenario == 3:  
        df1 = get_sample(df, rate , abnormal_count , normal_count , False)

    #more normal
    if scenario == 4:  
        df1 = get_sample(df, rate , abnormal_count, normal_count, False)
    return df1

def main():

    print("create base dataset begin")

    os.system("mkdir -p %s"%g_ds_base_path)
    os.system("mkdir -p %s"%g_ds_sample_path)

    create_dataset(0 , 1)
    create_dataset(1 , 0.2)
    create_dataset(2 , 0.2)
    create_dataset(3 , 0.2)
    print("create base dataset end")

    print("create sample begin")

    #scenario 1: diffrent dataset
    for sce in [ 1 , 2 , 3 , 4 ]:
        base_file = "%ss-0.csv"%g_ds_base_path 
        base = os.path.splitext(os.path.basename(base_file))[0]
        i = 0
        for rate in [ 0.5 , 0.1 , 0.5 , 1 ]:
            df = create_sample(base_file, rate ,  n = 500000 , scenario = sce )  
            df.to_csv(g_ds_sample_path + "%s-%d-%d.csv"%(base,sce,i))
            i = i + 1

    #scenario 2: more abnormal
    for sce in [ 1 , 2 , 3 , 4 ]:
        base_file = "%ss-1.csv"%g_ds_base_path
        base = os.path.splitext(os.path.basename(base_file))[0]
        i = 0
        for rate in [ 0.5 , 0.1 , 0.5 , 1 ]:
            df = create_sample(base_file, rate , n = 500000, scenario = sce )
            df.to_csv(g_ds_sample_path + "%s-%d-%d.csv"%(base,sce,i))
            i = i + 1

    #scenario 3: balance sample
    for sce in [ 1 , 2 , 3 , 4 ]:
        base_file = "%ss-2.csv"%g_ds_base_path
        base = os.path.splitext(os.path.basename(base_file))[0]
        i = 0
        for rate in [ 0.5 , 0.1 , 0.5 , 1 ]:
            df = create_sample(base_file, rate , n = 500000, scenario = sce )
            df.to_csv(g_ds_sample_path + "%s-%d-%d.csv"%(base,sce,i))
            i = i + 1

    #scenario 4: more normal 
    for sce in [ 1 , 2 , 3 , 4 ]:
        base_file = "%ss-3.csv"%g_ds_base_path
        base = os.path.splitext(os.path.basename(base_file))[0]
        i = 0
        for rate in [ 0.5 , 0.1 , 0.5 , 1 ]:
            df = create_sample(base_file, rate  , n = 500000, scenario = sce )
            df.to_csv(g_ds_sample_path + "%s-%d-%d.csv"%(base,sce,i))
            i = i + 1

    print("create sample end")

    #change to one-hot
    print("create one hot map begin")
    create_one_hot_map()

    csv_list = []
    get_csv_files(g_ds_sample_path,csv_list)
    df_all = pd.DataFrame()
    for fi in csv_list:
        print("add one hot to sample file ", fi)
        df = pd.read_csv(fi,index_col=0)
        df = create_one_hot_column( df )
        df.to_csv(fi)

    print("create one hot map finished")

if __name__ == "__main__":
    main()
