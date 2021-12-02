import pandas as pd
import os,json
from sklearn.preprocessing import OneHotEncoder

g_ds_root = "/data/paper/"
g_ds_apo_raw_file_path = g_ds_root + "raw/iot123/Malware-Project/BigDataset/IoTScenarios/"
g_ds_apo_raw_csv_path = g_ds_root + "common/apo123/raw_csv/"
g_ds_apo_common_csv_path = g_ds_root + "common/apo123/common/"

g_ds_ton_raw_csv_path = g_ds_root + "raw/ton/Train_Test_Network_dataset/"
g_ds_ton_common_csv_path = g_ds_root + "common/ton/"

g_ds_base_path = g_ds_root + "base/"
g_ds_sample_path = g_ds_root + "sample/"

g_model_path = g_ds_root + "model/"
g_analyse_path = g_ds_root + "analyse/"

g_one_hot_file = g_ds_sample_path+ "/one_hot.json"
g_one_hot_colums = ['conn_state','service','proto']


#read all csv files
def get_csv_files(filepath,file_list=[]):
    files = os.listdir(filepath)
    for fi in files:
        fi_d = os.path.join(filepath,fi)
        if os.path.isdir(fi_d):
            get_csv_files(fi_d,file_list)
        else:
            if fi_d.endswith("csv"):
                file_list.append(fi_d)

#get one hot code for a column
def get_one_hot_code(df,key):
    key_list = df[key].unique()
    key_list.sort()
    key_list = key_list.reshape(len(key_list),1)
    ret = {}
    enc = OneHotEncoder()
    one_hot = enc.fit_transform(key_list).toarray()
    for raw_code,enc_code in zip(key_list,one_hot):
        ret[raw_code[0]] = enc_code.tolist()
    return ret

#create one hot map file ( ran after samples generated )
def create_one_hot_map(path = g_ds_sample_path):
    csv_list = []
    get_csv_files(path,csv_list)
    df_all = pd.DataFrame()
    for i in csv_list:
        df = pd.read_csv(i,index_col=0)
        df_all = pd.concat([df_all,df],ignore_index=True)

    one_hot = {}
    df_test = df_all.copy( deep = True )
    for key in g_one_hot_colums:
        one_hot[key] = get_one_hot_code(df_test,key)

    with open(g_one_hot_file,"w") as fp:
        fp.write(json.dumps(one_hot,indent=4))

    print("one-hot map file is created at ",g_one_hot_file)

#add one hot column to DataFrame
def create_one_hot_column(df):

    with open(g_one_hot_file,"r") as fp:
        one_hot_map = json.loads(fp.read())
    df_ret = df.copy(deep = True)
    for prefix in g_one_hot_colums:
        key_list = list(one_hot_map[prefix].keys())
        for key in key_list:
            mask = df_ret[prefix]== key
            en_code = one_hot_map[prefix][key]
            for i in range(len(key_list)):
                head_key = '%s_%s'%(prefix,key_list[i])
                df_ret.loc[mask,head_key] = en_code[i]
    return df_ret

#check if sample data qualified. 
#1.At least, samples contain two type lables (0 and 1)
#2.At least, sample count more than 100
def is_sample_qualified(df_sample):
    if len(df_sample['label'].unique()) <= 1 or df_sample.shape[0] < 100:
        return False
    return True

#preprocess input sample data
#1.transfer object type to one-hot code
#2.shuffle sample
def preprocess(df_sample):
    df_temp = df_sample.copy( deep = True )
    df_ret = pd.DataFrame()
    for key in df_sample:
        if not key in ['label','type','dataset'] and not key in g_one_hot_colums:
            df_ret[key] = df_temp[key]
    df_ret['label'] = df_temp['label'].map(lambda x: 1 if x.lower() != 'benign' else 0)
    
    for key in ['src_port','dst_port','duration','src_bytes','dst_bytes','missed_bytes','src_pkts','dst_pkts','dst_ip_bytes']:
        if not df_ret[key].std() == 0 :
            df_ret[key] = (df_ret[key] - df_ret[key].mean())/ df_ret[key].std()

    df_ret = df_ret.sample(frac=1).reset_index(drop=True)
    return df_ret

def main():
    sample_file = "/data/dataset/apo123/common/CTU-IoT-Malware-Capture-34-1.csv"
    #sample_file = "/data/dataset/ton/common/ton.csv"
    df = pd.read_csv(sample_file,index_col=0)
    df_all = preprocess(df)
    if is_sample_qualified(df_all):
        print("sample is qualified",df_all.shape)
    else:
        print("sample is NOT qualified",df_all.shape)

if __name__ == "__main__":
    main()
