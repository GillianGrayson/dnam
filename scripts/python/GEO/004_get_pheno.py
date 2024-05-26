import GEOparse
import pandas as pd
import pathlib
import re
from scripts.python.GEO.routines import get_gse_gsm_info, process_characteristics_ch1


gse = 'GSE87648'
datasets_info = pd.read_excel(f"D:/YandexDisk/Work/pydnameth/datasets/datasets.xlsx", index_col='dataset')
gpl = datasets_info.loc[gse, 'platform']

characteristics_ch1_regex_findall = ';*([a-zA-Z0-9\^\/\=\-\:\,\.\s_\(\)]+): '
characteristics_ch1_regex_split = '(;*[a-zA-Z0-9\^\/\=\-\,\:\.\s_\(\)]+: )'

path = "D:/YandexDisk/Work/pydnameth/datasets"

gsm_df, gse_gsms_dict = get_gse_gsm_info(f"{path}/GEO", gpl)

pathlib.Path(f"{path}/{gpl}/{gse}/raw/GEO").mkdir(parents=True, exist_ok=True)
gsms = gse_gsms_dict[gse]
gse_df_1 = gsm_df.loc[gsm_df.index.isin(gsms), :]

while True:
    try:
        gse_data = GEOparse.get_GEO(geo=gse, destdir=f"{path}/{gpl}/{gse}/raw/GEO", include_data=False, how="quick", silent=False)
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
    gse_df_2 = gse_df_2.loc[(gse_df_2['platform_id'] == gpl), :]
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
    print(f"GEOmetadb: \n {chars_df_1}")
    print(f"GEOparse: \n {chars_df_2}")

if not gse_df['supplementary_file'].isnull().all():
    supp_files_split = gse_df['supplementary_file'].str.split('[,;]\s*', expand=True, regex=True)
    if supp_files_split.shape[1] == 2:
        gse_df[['supplementary_file_1', 'supplementary_file_2']] = supp_files_split
        supp_details = gse_df['supplementary_file_1'].str.findall('(?:.*\/)(.*)(?:_\w*.\..*\..*)').explode().str.split('_', expand=True)
        if supp_details.shape[1] == 3:
            gse_df[['Sample_Name', 'Sentrix_ID', 'Sentrix_Position']] = supp_details
        else:
            n_fields = supp_details.shape[1]
            fields_name = [f"raw_idat_{x}" for x in range(n_fields)]
            gse_df[fields_name] = supp_details

gse_df.to_excel(f"{path}/{gpl}/{gse}/pheno11.xlsx", index=True)

