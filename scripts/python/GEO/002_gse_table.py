import GEOparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from distutils.dir_util import copy_tree
from pathlib import Path
import pathlib
import pickle
import os
import re

def process_characteristics_ch1(gse_df):
    to_remove = [';', ': ']
    remover = re.compile('|'.join(map(re.escape, to_remove)))
    for gsm, row in gse_df.iterrows():
        tmp = re.split(r'(;*[a-zA-Z0-9 _]+: )', row['characteristics_ch1'])
        tmp = [x for x in tmp if x]
        pairs = dict(zip(tmp[::2], tmp[1::2]))
        for char_raw, value in pairs.items():
            char = remover.sub('', char_raw)
            gse_df.at[gsm, char] = value


GPL = 'GPL13534'
n_gses = 200

path = "E:/YandexDisk/Work/pydnameth/datasets/GEO"

if Path(f"{path}/{GPL}_gsm.pkl").is_file():
    gsm_df = pd.read_pickle(f"{path}/{GPL}_gsm.pkl")
else:
    gsm_df = pd.read_excel(f"{path}/{GPL}_gsm.xlsx", index_col='gsm')
    gsm_df.to_pickle(f"{path}/{GPL}_gsm.pkl")

pathlib.Path(f"{path}/{GPL}").mkdir(parents=True, exist_ok=True)
if Path(f"{path}/{GPL}/gse_gms_dict.pkl").is_file():
    f = open(f"{path}/{GPL}/gse_gms_dict.pkl", 'rb')
    gse_gsms_dict = pickle.load(f)
    f.close()
else:
    gse_gsms_dict = {}
    for gsm, row in gsm_df.iterrows():
        gses_i = row['series_id'].split(',')
        for gse in gses_i:
            if gse not in gse_gsms_dict:
                gse_gsms_dict[gse] = [gsm]
            else:
                gse_gsms_dict[gse].append(gsm)
    f = open(f"{path}/{GPL}/gse_gms_dict.pkl", 'wb')
    pickle.dump(gse_gsms_dict, f, pickle.HIGHEST_PROTOCOL)
    f.close()

gses = sorted(gse_gsms_dict.keys(),  key=lambda s: len(gse_gsms_dict.get(s)),  reverse=True)
gses_df = pd.DataFrame(index=gses, columns=['Count', 'characteristics_ch1', 'source_name_ch1', 'raw_files_exist'])
gses_df.index.name = 'GSE'

for gse_id, gse in enumerate(gses[0:n_gses]):
    if gse_id < 0:
        continue
    print(f"{gse_id}: {gse}")
    pathlib.Path(f"{path}/{GPL}/{gse_id}_{gse}").mkdir(parents=True, exist_ok=True)
    gsms_i = gse_gsms_dict[gse]
    gse_df_1 = gsm_df.loc[gsm_df.index.isin(gsms_i), :]

    while True:
        try:
            gse_data = GEOparse.get_GEO(geo=gse, destdir=f"{path}/{GPL}/{gse_id}_{gse}", include_data=False, how="quick", silent=True)
        except ValueError:
            continue
        except ConnectionError:
            continue
        except IOError:
            continue
        break
    gse_df_2 = gse_data.phenotype_data
    GEOparse_data_empy = gse_df_2.empty
    if not GEOparse_data_empy:
        gse_df_2.replace('NONE', pd.NA, inplace=True)
        gse_df_2.index.name = 'gsm'
        gse_df_2 = gse_df_2.loc[(gse_df_2['platform_id'] == GPL), :]
        is_index_equal = set(gse_df_1.index) == set(gse_df_2.index)
        gses_df.at[gse, 'Count'] = gse_df_2.shape[0]
        if not is_index_equal:
            print("Index is not equal in GEOmetadb and GEOparse")
            print(f"gse_df_2 full: {gse_df_2.shape[0]}")
            print(f"gse_df_1 full: {gse_df_1.shape[0]}")
            unique_chars = set()
        else:
            print("Index is equal")
            gse_df_1 = gse_df_1.loc[gse_df_2.index, :]
            gse_df_2 = pd.merge(gse_df_2, gse_df_1['characteristics_ch1'], left_index=True, right_index=True)
            unique_chars = set.union(*gse_df_2['characteristics_ch1'].str.findall('([a-zA-Z0-9 _]+):').apply(set).to_list())
            process_characteristics_ch1(gse_df_2)
    else:
        print("No data from GEOparse")
        gse_df_2 = gse_df_1
        unique_chars = set.union(*gse_df_2['characteristics_ch1'].str.findall('([a-zA-Z0-9 _]+):').apply(set).to_list())
        process_characteristics_ch1(gse_df_2)

    chars_cols = gse_df_2.columns.values[gse_df_2.columns.str.startswith('characteristics_ch1.')]
    r = re.compile(r"characteristics_ch1.\d*.(.*)")
    remaining_chars = [r.findall(x)[0] for x in chars_cols]
    gses_df.at[gse, 'characteristics_ch1'] = unique_chars.union(set(remaining_chars))
    gses_df.at[gse, 'source_name_ch1'] = gse_df_2['source_name_ch1'].unique()

    if not gse_df_2['supplementary_file'].isnull().all():
        gses_df.at[gse, 'raw_files_exist'] = True
        if len(gse_df_2['supplementary_file'].unique()) == len(gse_df_2.index):
            gses_df.at[gse, 'raw_files_for_all'] = True
        else:
            gses_df.at[gse, 'raw_files_for_all'] = False
        gse_df_2[['supplementary_file_1', 'supplementary_file_2']] = gse_df_2['supplementary_file'].str.split(',\s*', expand=True, regex=True)
        tmp = gse_df_2['supplementary_file_1'].str.findall('(?:.*\/)(.*)(?:_\w*.\..*\..*)').explode()
        gse_df_2[['Sample_Name', 'Sentrix_ID', 'Sentrix_Position']] = tmp.str.split('_', expand=True)
    else:
        gses_df.at[gse, 'raw_files_exist'] = False

    if GEOparse_data_empy:
        gse_df_2.to_excel(f"{path}/{GPL}/{gse_id}_{gse}/gse_GEOmetadb.xlsx", index=True)
    else:
        gse_df_2.to_excel(f"{path}/{GPL}/{gse_id}_{gse}/gse_GEOparse.xlsx", index=True)

gses_df.to_excel(f"{path}/{GPL}/gses.xlsx", index=True)
