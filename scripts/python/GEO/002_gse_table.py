import GEOparse
import pandas as pd
from pathlib import Path
import pathlib
import pickle
import re


def process_characteristics_ch1(df, regex_split):
    to_remove = [';', ': ']
    remover = re.compile('|'.join(map(re.escape, to_remove)))
    for gsm, row in df.iterrows():
        tmp = re.split(regex_split, row['characteristics_ch1'])
        tmp = [x for x in tmp if x]
        pairs = dict(zip(tmp[::2], tmp[1::2]))
        for char_raw, value in pairs.items():
            char = remover.sub('', char_raw)
            df.at[gsm, char] = value


GPL = 'GPL21145'
n_gses = 300

characteristics_ch1_regex_findall = ';*([a-zA-Z0-9\^\/\=\-\:\,\.\s_\(\)]+): '
characteristics_ch1_regex_split = '(;*[a-zA-Z0-9\^\/\=\-\,\:\.\s_\(\)]+: )'

path = "D:/YandexDisk/Work/pydnameth/datasets/GEO"

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
gses_df = pd.DataFrame(index=gses, columns=['raw_files_exist'])
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
    if gse_df_2.empty:
        process_type = 'GEOmetadb'
    else:
        gse_df_2.index.name = 'gsm'
        gse_df_2.replace('NONE', pd.NA, inplace=True)
        gse_df_2 = gse_df_2.loc[(gse_df_2['platform_id'] == GPL), :]
        is_index_equal = set(gse_df_1.index) == set(gse_df_2.index)
        if is_index_equal:
            process_type = 'Common'
        else:
            print(f"GEOmetadb: {gse_df_1.shape[0]}")
            print(f"GEOparse: {gse_df_2.shape[0]}")
            if gse_df_2.shape[0] > gse_df_1.shape[0]:
                process_type = 'GEOparse'
            else:
                process_type = 'GEOmetadb'
    print(f"process_type: {process_type}")

    gses_df.at[gse, 'GEOmetadb'] = gse_df_1.shape[0]
    gses_df.at[gse, 'GEOparse'] = gse_df_2.shape[0]

    if process_type == 'Common':
        gse_df_1 = gse_df_1.loc[gse_df_2.index, :]
        gse_df = pd.merge(gse_df_2, gse_df_1['characteristics_ch1'], left_index=True, right_index=True)
        chars_df_1 = set.union(*gse_df['characteristics_ch1'].str.findall(characteristics_ch1_regex_findall).apply(set).to_list())
        process_characteristics_ch1(gse_df, characteristics_ch1_regex_split)
    elif process_type == 'GEOmetadb':
        gse_df = gse_df_1
        chars_df_1 = set.union(*gse_df['characteristics_ch1'].str.findall(characteristics_ch1_regex_findall).apply(set).to_list())
        process_characteristics_ch1(gse_df, characteristics_ch1_regex_split)
    elif process_type == 'GEOparse':
        gse_df = gse_df_2.copy()
        chars_df_1 = set()
    else:
        raise ValueError(f"Unsupported process_type")

    if process_type in ['GEOparse', 'Common']:
        chars_cols = gse_df.columns.values[gse_df.columns.str.startswith('characteristics_ch1.')]
        r = re.compile(r"characteristics_ch1.\d*.(.*)")
        chars_df_2 = set([r.findall(x)[0] for x in chars_cols])
    else:
        chars_df_2 = set()

    chars_all = chars_df_1.union(chars_df_2)

    if chars_df_2 != chars_df_1:
        print(f"Chars from GEOmetadb ({len(chars_df_1)}) and GEOparse ({len(chars_df_2)}) differs!")
        gses_df.at[gse, 'characteristics_ch1_differs'] = True
    else:
        gses_df.at[gse, 'characteristics_ch1_differs'] = False

    gses_df.at[gse, 'n_characteristics_ch1_GEOmetadb'] = len(chars_df_1)
    gses_df.at[gse, 'n_characteristics_ch1_GEOparse'] = len(chars_df_2)
    gses_df.at[gse, 'process_type'] = process_type
    gses_df.at[gse, 'Count'] = gse_df.shape[0]
    gses_df.at[gse, 'characteristics_ch1'] = str(chars_all)
    gses_df.at[gse, 'source_name_ch1'] = gse_df['source_name_ch1'].unique()

    if not gse_df['supplementary_file'].isnull().all():
        gses_df.at[gse, 'raw_files_exist'] = True
        if len(gse_df['supplementary_file'].unique()) == len(gse_df.index):
            gses_df.at[gse, 'raw_files_for_all'] = True
        else:
            gses_df.at[gse, 'raw_files_for_all'] = False
        supp_files_split = gse_df['supplementary_file'].str.split('[,;]\s*', expand=True, regex=True)
        if supp_files_split.shape[1] == 2:
            gse_df[['supplementary_file_1', 'supplementary_file_2']] = supp_files_split
            supp_details = gse_df['supplementary_file_1'].str.findall('(?:.*\/)(.*)(?:_\w*.\..*\..*)').explode().str.split('_', expand=True)
            if supp_details.shape[1] == 3:
                gse_df[['Sample_Name', 'Sentrix_ID', 'Sentrix_Position']] = supp_details
    else:
        gses_df.at[gse, 'raw_files_exist'] = False

    gse_df.to_excel(f"{path}/{GPL}/{gse_id}_{gse}/{process_type}.xlsx", index=True)

gses_df.to_excel(f"{path}/{GPL}/gses.xlsx", index=True)
