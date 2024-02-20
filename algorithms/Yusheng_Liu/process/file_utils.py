import os
import csv
import json
import lmdb
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd

from typing import Sequence


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.__str__()
        else:
            return super(MyEncoder, self).default(obj)


def find_all_files(root, suffix=None):
    res = list()
    for root, _, files in os.walk(root):
        for f in files:
            if suffix is not None and not f.endswith(suffix):
                continue
            res.append(os.path.join(root, f))
    return res


def load_df(file, rename_dict=None, sheet_name=None):
    if file.endswith('.csv'):
        df = pd.read_csv(file, encoding='utf-8')
    elif file.endswith('.xlsx') or file.endswith('.xls'):
        df = pd.read_excel(file, sheet_name=0 if sheet_name is None else sheet_name, encoding='utf-8')
    else:
        print('Bad file %s with invalid format, please check in manual!' % file)
        return None

    if rename_dict is not None:
        df = df.rename(columns=rename_dict)

    df.drop_duplicates(inplace=True)  # 删除重复行
    df.reset_index(drop=True, inplace=True)  # 更新index
    return df


def read_csv(csv_path):
    with open(csv_path, 'r', newline='') as file:
        csv_reader = csv.reader(file, delimiter=',')
        rows = [row for row in csv_reader]
    return rows


def save_csv(file, data, name=None):
    if not os.path.exists(file):
        os.mknod(file)

    data = pd.DataFrame(columns=name, data=data)
    data.drop_duplicates(inplace=True)
    data.to_csv(file, index=False, encoding='utf-8-sig')


def write_csv(csv_name, content, mul=True, mod="w"):
    """write list to .csv file."""
    with open(csv_name,mod,newline="") as myfile:
        writer = csv.writer(myfile)
        if mul:
            writer.writerows(content)
        else:
            writer.writerow(content)


def load_json(json_file):
    # 将文件中的数据读出来
    f = open(json_file, 'r')
    file_data = json.load(f)
    f.close()
    return file_data


def save_json(json_file, dict_data):
    # 将字典保存在filename文件中，并保存在directory文件夹中
    directory = os.path.dirname(json_file)  # 有可能直接给文件名，没有文件夹
    if (directory != '') and (not os.path.exists(directory)):
        os.makedirs(directory)
    f = open(json_file, 'wt')
    json.dump(dict_data, f, cls=MyEncoder, sort_keys=True, indent=4)
    f.close()


def read_txt(txt_file):
    txt_lines = []
    with open(txt_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            txt_lines.append(line)
    return txt_lines


def write_txt(txt_file, data):
    with open(txt_file, 'w') as f:
        for item in data:
            s = item + '\n'
            f.write(s)


def split_filename(file_name: str) -> List[str]:
    """
    Split the filename of medical image.
    :param file_name:
    :return:
    List[series_uid, suffix]
    """
    if file_name.endswith('.nii.gz'):
        str_list = file_name.split('.nii.gz')
    elif file_name.endswith('.mha'):
        str_list = file_name.split('.mha')
    elif file_name.endswith('.mhd'):
        str_list = file_name.split('.mhd')
    elif file_name.endswith('.npz'):
        str_list = file_name.split('.npz')
    elif file_name.endswith('.npy'):
        str_list = file_name.split('.npy')
    else:
        str_list = [file_name, '']

    return str_list


def merge_db(source_db_file: Sequence[str], out_db: str):
    """
    Merge multi database file.
    :param source_db_file: List of db file.
    :param out_db: output db in place.
    :return:
    """
    out_env = lmdb.open(out_db, map_size=int(1e9))
    txn_out = out_env.begin(write=True)
    for db_file in source_db_file:
        db_env = lmdb.open(db_file, map_size=int(1e9))
        txn_db = db_env.begin()
        for key, value in txn_db.cursor():
            txn_out.put(key=key, value=value)
        db_env.close()

    txn_out.commit()
    out_env.close()


def merge_txt(source_txt_file: Sequence[str], out_txt: str):
    """
    Merge multi txt file.
    :param source_txt_file: List of txt file.
    :param out_txt: out txt file in place.
    :return:
    """
    out_content = []
    for txt_file in source_txt_file:
        content = read_txt(txt_file)
        out_content.extend(content)
    out_content = set(out_content)

    write_txt(out_txt, out_content)


def merge_csv(source_csv_file: Sequence[str], out_csv: str):
    """
    Merge multi csv file.
    :param source_csv_file: List of csv file.
    :param out_csv: out csv file in place.
    :return:
    """
    df_all = []
    for csv_file in source_csv_file:
        df = pd.read_csv(csv_file, encoding='utf-8')
        df_all.append(df)
    df_all = pd.concat(df_all, axis=0, ignore_index=True)
    df_all.to_csv(out_csv, encoding='utf-8')


class DataBaseUtils(object):
    def __init__(self):
        super(DataBaseUtils, self).__init__()

    @staticmethod
    def creat_db(db_dir):
        env = lmdb.open(db_dir, map_size=int(1e9))
        txn = env.begin(write=True)
        txn.commit()
        env.close()

    @staticmethod
    def update_record_in_db(db_dir, idx, data_dict):
        env = lmdb.open(db_dir, map_size=int(1e9))
        txn = env.begin(write=True)
        txn.put(str(idx).encode(), value=json.dumps(data_dict, cls=MyEncoder).encode())
        txn.commit()
        env.close()

    @staticmethod
    def delete_record_in_db(db_dir, idx):
        env = lmdb.open(db_dir, map_size=int(1e9))
        txn = env.begin(write=True)
        txn.delete(str(idx).encode())
        txn.commit()
        env.close()

    @staticmethod
    def get_record_in_db(db_dir, idx):
        env = lmdb.open(db_dir, map_size=int(1e9))
        txn = env.begin()
        value = txn.get(str(idx).encode())
        env.close()
        if value is None:
            return None
        value = str(value, encoding='utf-8')
        data_dict = json.loads(value)

        return data_dict

    @staticmethod
    def read_records_in_db(db_dir):
        env = lmdb.open(db_dir, map_size=int(1e9))
        txn = env.begin()
        out_records = dict()
        for key, value in txn.cursor():
            key = str(key, encoding='utf-8')
            value = str(value, encoding='utf-8')

            label_info = json.loads(value)
            out_records[key] = label_info
        env.close()

        return out_records

    @staticmethod
    def write_records_in_db(db_dir, in_records):
        env = lmdb.open(db_dir, map_size=int(1e9))
        txn = env.begin(write=True)
        for key, value in in_records.items():
            txn.put(str(key).encode(), value=json.dumps(value, cls=MyEncoder).encode())
        txn.commit()
        env.close()