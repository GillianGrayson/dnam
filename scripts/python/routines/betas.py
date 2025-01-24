import pandas as pd


def betas_drop_na(betas: pd.DataFrame, is_print_na_pairs=False):
    na_cols = betas.columns[betas.isna().any()].tolist()
    print(f"Number of CpGs before drop_na: {betas.shape[1]}")
    if len(na_cols) > 0:
        if is_print_na_pairs:
            print(f"CpGs with NaNs: {na_cols}")
            s = betas.stack(dropna=False)
            na_pairs = [list(x) for x in s.index[s.isna()]]
            print(*na_pairs, sep='\n')
    betas.dropna(axis='columns', how='any', inplace=True)
    print(f"Number of CpGs after drop_na: {betas.shape[1]}")
    return betas
