# -*- coding: utf-8 -*-
"""
    This script is used to convert dataset apo123 to common csv fromat, including 2 steps:

    1. Convert apo123 to csv format ( all feilds ) from raw files.
 
    2. Convert common csv format ( 13 feilds )

    Edit by Jim Xie (xiewenwei@sina.com)  2021/11/28
"""

import pandas as pd
import os
from common import get_csv_files,g_ds_apo_raw_csv_path,g_ds_apo_raw_file_path,g_ds_apo_common_csv_path

def get_file_type(input_file):
    tmp = input_file
    pos = tmp.find("/CTU-")
    tmp = tmp[pos+1:]
    pos = tmp.find(".csv")
    dataType = tmp[0:pos]
    return dataType

def get_labeled_files(filepath,file_list=[]):
    files = os.listdir(filepath)
    for fi in files:
        fi_d = os.path.join(filepath,fi)
        if os.path.isdir(fi_d):
            get_labeled_files(fi_d,file_list)
        else:
            if fi_d.endswith("labeled"):
                file_list.append(fi_d)

def log2csv(input_file,output_file):
    csv_data = []
    with open(input_file) as fp:
        data = fp.readline()
        while data:
            if data[0] == '#':
                if data.find("#fields") ==0:
                    data = data.replace("   ","\t")
                    data = data.replace("#fields	","").strip("\n")
                    header = ",".join(data.split("\t"))
                    csv_data.append(header)
            else:
                data = data.replace("   ","\t")
                data = ",".join(data.strip("\n").split("\t"))
                csv_data.append(data)
            data = fp.readline()
    #output_file = input_file.replace("labeled","csv")
    with open(output_file,"w") as fp:
        for data in csv_data:
            fp.write(data+"\n")

def get_file_name(input_file):
    tmp = input_file
    pos = tmp.find("/CTU-")
    tmp = tmp[pos+1:]
    pos = tmp.find("/")
    dataType = tmp[0:pos]
    return dataType

def to_common_format(file_name):
    df_apo = pd.read_csv(file_name)
    #df_apo = df_apo[df_apo['duration'] != '-']
    #df_apo[['id.orig_p','id.resp_p','orig_bytes','resp_bytes','orig_pkts','resp_pkts','resp_ip_bytes']]

    mask = (df_apo['label']=='Benign')|(df_apo['label']=='benign')
    df_apo.loc[mask, 'label'] = 'benign'
    mask = (df_apo['label']!='Benign')&(df_apo['label']!='benign')
    df_apo.loc[mask, 'label'] = 'abnormal'
    mask = df_apo['duration']=='-'
    df_apo.loc[mask, 'duration'] = 0
    df_apo.loc[mask, 'orig_bytes'] = 0
    df_apo.loc[mask, 'resp_bytes'] = 0
    df_apo.loc[mask, 'missed_bytes'] = 0
    df_apo.loc[mask, 'orig_pkts'] = 0
    df_apo.loc[mask, 'orig_ip_bytes'] = 0
    df_apo.loc[mask, 'resp_pkts'] = 0
    df_apo.loc[mask, 'resp_ip_bytes'] = 0

    df = pd.DataFrame()
    df['proto'] = df_apo['proto'].astype('category')
    df['service'] = df_apo['service'].astype('category')
    df['duration'] = df_apo['duration'].astype("float")
    df['missed_bytes'] = df_apo['missed_bytes'].astype('int64')
    df['conn_state'] = df_apo['conn_state'].astype('category')
    df['src_port'] = df_apo['id.orig_p'].astype('int64')
    df['dst_port'] = df_apo['id.resp_p'].astype('int64')
    df['src_bytes'] = df_apo['orig_bytes'].astype('int64')
    df['dst_bytes'] = df_apo['resp_bytes'].astype('int64')
    df['src_pkts'] = df_apo['orig_pkts'].astype('int64')
    df['dst_pkts'] = df_apo['resp_pkts'].astype('int64')
    df['dst_ip_bytes'] = df_apo['resp_ip_bytes'].astype('int64')
    df['label'] = df_apo['label'].astype('category')
    df['type'] = df_apo['detailed-label'].astype('category')
    df['dataset'] = get_file_type(file_name)

    return df

def main():

    '''
    print("convert apo123 to raw csv foramt begin ")

    os.system("mkdir -p %s"%g_ds_apo_raw_csv_path)
    os.system("mkdir -p %s"%g_ds_apo_common_csv_path)

    file_list = []
    get_labeled_files(g_ds_apo_raw_file_path,file_list)
    for f in file_list:
        csv_file = "%s/%s.csv"%(g_ds_apo_raw_csv_path,get_file_name(f))
        print("coverting...",csv_file,f)
        log2csv(f,csv_file)

    print("convert apo123 to raw csv foramt end ")
    '''
    print("convert apo123 to common csv foramt begin ")

    file_list = []
    get_csv_files(g_ds_apo_raw_csv_path,file_list)
    for f in file_list:
        print("coverting...",f)
        df = to_common_format(f) # assuming the file contains a header
        df.to_csv("%s/%s.csv"%(g_ds_apo_common_csv_path,get_file_type(f)))

    print("convert apo123 to common csv foramt end ")

if __name__ == "__main__":
    main()
