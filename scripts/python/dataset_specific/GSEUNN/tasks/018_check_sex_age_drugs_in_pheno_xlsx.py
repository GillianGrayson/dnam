import pandas as pd
import numpy as np
from scripts.python.pheno.datasets.filter import filter_pheno
from scripts.python.pheno.datasets.features import get_column_name, get_status_dict, get_sex_dict
from scripts.python.routines.manifest import get_manifest




dataset = "GSEUNN"
path = f"E:/YandexDisk/Work/pydnameth/datasets"
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')
platform = datasets_info.loc[dataset, 'platform']
manifest = get_manifest(platform)
status_col = get_column_name(dataset, 'Status').replace(' ','_')
age_col = get_column_name(dataset, 'Age').replace(' ','_')
sex_col = get_column_name(dataset, 'Sex').replace(' ','_')
status_dict = get_status_dict(dataset)
status_passed_fields = status_dict['Control'] + status_dict['Case']
sex_dict = get_sex_dict(dataset)
continuous_vars = {}
categorical_vars = {status_col: [x.column for x in status_passed_fields], sex_col: list(sex_dict.values())}
pheno = pd.read_excel(f"{path}/{platform}/{dataset}/pheno_xtd.xlsx", index_col='subject_id')
pheno.to_pickle(f"{path}/{platform}/{dataset}/pheno_xtd.pkl")
pheno_old = pheno.copy()
pheno = filter_pheno(dataset, pheno, continuous_vars, categorical_vars)

ages_sexes = pd.read_excel(f"{path}/{platform}/{dataset}/data/age_sex_L_H_A_Q_I_S_T.xlsx", index_col='Code')

for index, row in pheno.iterrows():
    curr_sex = row['Sex']
    real_sex = ages_sexes.at[row['ID'], 'Sex']
    curr_age = row['Age']
    real_age = ages_sexes.at[row['ID'], 'Age']
    if curr_sex != real_sex:
        print(f"Wrong sex: {row['ID']}")
    if abs(real_age - curr_age) > 1e-1:
        print(f"Wrong age for {row['ID']}: diff = {abs(real_age - curr_age)}")

diseases = pd.read_excel(f"{path}/{platform}/{dataset}/data/pheno/fixes_from_EK.xlsx", index_col='ID', sheet_name='Diseases')

for col in diseases.columns:
    if col not in pheno.columns:
        if len(set(diseases.loc[pheno['ID'], col].values)) > 2:
            print(f"There is new disease: {col}")
    else:
        new = diseases.loc[pheno['ID'], col].values
        old = pheno.loc[:, col].values
        check = (new == old).all()
        if not check:
            print(f"There is updates for the disease: {col}")
            diff_ids = pheno['ID'].values[np.where(new != old)]
            print(f"{diff_ids}")
        pheno.loc[:, col] = diseases.loc[pheno['ID'], col].values

pheno_old_add = pheno_old.loc[~pheno_old.index.isin(pheno.index), :]
pheno_all = pheno.append(pheno_old_add, verify_integrity=True)

pheno_all.to_pickle(f"{path}/{platform}/{dataset}/pheno_xtd_tmp.pkl")
pheno_all.to_excel(f"{path}/{platform}/{dataset}/pheno_xtd_tmp.xlsx", index=True)