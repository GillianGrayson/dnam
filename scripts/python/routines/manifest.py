import pandas as pd
import os
import re


def get_genes_list(df: pd.DataFrame, col: str, emptys, separators):
    genes_raw = df.loc[:, col].values.astype(str)
    genes_all = set()
    for genes_row in genes_raw:
        if genes_row not in emptys:
            genes = set(re.split(separators, genes_row))
            genes_all.update(genes)
    return list(genes_all)


def process_str_elem(x, delimiter: str = ';', missed: str = 'non-genic'):
    if isinstance(x, str):
        elems = x.split(';')
        elems = list(set(elems))
        elems = delimiter.join(elems)
    else:
        elems = missed
    return elems


def get_manifest(platform="GPL13534", path="E:/YandexDisk/Work/pydnameth/datasets"):

    fn_pkl = f"{path}/{platform}/manifest/manifest.pkl"
    if os.path.isfile(fn_pkl):
        manifest = pd.read_pickle(fn_pkl)
    else:
        fn = f"{path}/{platform}/manifest/manifest.xlsx"
        manifest = pd.read_excel(fn, index_col="CpG")
        manifest['Gene'] = manifest['Gene'].apply(process_str_elem)
        manifest['UCSC_RefGene_Group'] = manifest['UCSC_RefGene_Group'].apply(process_str_elem)
        manifest.to_pickle(fn_pkl)

    return manifest
