# -*- coding: utf-8 -*-
"""
    This script is used to convert dataset NoT to common csv fromat ( 13 feilds )

    Edit by Jim Xie (xiewenwei@sina.com) 2021/11/28
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time,json,os
from common import g_ds_ton_common_csv_path,g_ds_ton_raw_csv_path

def main():
    raw_file = "%s/Train_Test_Network.csv"%g_ds_ton_raw_csv_path
    common_file = "%s/ton.csv"%g_ds_ton_common_csv_path
    os.system("mkdir -p %s"%g_ds_ton_common_csv_path)

    df_ton = pd.read_csv(raw_file)
    df_ton['label'] = df_ton['label'].map(lambda x: 'benign' if x == 0  else 'abnormal')
    df = pd.DataFrame()
    df['src_port'] = df_ton['src_port'].astype('int64')
    df['dst_port'] = df_ton['dst_port'].astype('int64')
    df['proto'] = df_ton['proto'].astype('category')
    df['service'] = df_ton['service'].astype('category')
    df['duration'] = df_ton['duration'].astype("float")
    df['src_bytes'] = df_ton['src_bytes'].astype('int64')
    df['dst_bytes'] = df_ton['dst_bytes'].astype('int64')
    df['conn_state'] = df_ton['conn_state'].astype('category')
    df['missed_bytes'] = df_ton['missed_bytes'].astype('int64')
    df['src_pkts'] = df_ton['src_pkts'].astype('int64')
    df['dst_pkts'] = df_ton['dst_pkts'].astype('int64')
    df['dst_ip_bytes'] = df_ton['dst_ip_bytes'].astype('int64')
    df['label'] = df_ton['label'].astype('category')
    df['type'] = df_ton['type'].astype('category')
    df['dataset'] = "ToN"
    df.to_csv(common_file)
    print("raw file:",raw_file)
    print("common file:",common_file)

if __name__ == "__main__":
    main()
