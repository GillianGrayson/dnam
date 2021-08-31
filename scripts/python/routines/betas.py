import pandas as pd


def betas_drop_na(betas: pd.DataFrame):
    na_cols = betas.columns[betas.isna().any()].tolist()
    if len(na_cols) > 0:
        print(f"CpGs with NaNs: {na_cols}")
        s = betas.stack(dropna=False)
        na_pairs = [list(x) for x in s.index[s.isna()]]
        print(*na_pairs, sep='\n')
    betas.dropna(axis='columns', how='any', inplace=True)
    return betas
