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


def get_gse_gsm_info(path, gpl):
    if Path(f"{path}/{gpl}_gsm.pkl").is_file():
        gsm_df = pd.read_pickle(f"{path}/{gpl}_gsm.pkl")
    else:
        gsm_df = pd.read_excel(f"{path}/{gpl}_gsm.xlsx", index_col='gsm')
        gsm_df.to_pickle(f"{path}/{gpl}_gsm.pkl")

    pathlib.Path(f"{path}/{gpl}/").mkdir(parents=True, exist_ok=True)
    if Path(f"{path}/{gpl}/gse_gms_dict.pkl").is_file():
        f = open(f"{path}/{gpl}/gse_gms_dict.pkl", 'rb')
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
        f = open(f"{path}/{gpl}/gse_gms_dict.pkl", 'wb')
        pickle.dump(gse_gsms_dict, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    return gsm_df, gse_gsms_dict
