import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.bar import add_bar_trace
from scripts.python.routines.plot.layout import add_layout
from scripts.python.routines.manifest import get_manifest
from scripts.python.pheno.datasets.filter import filter_pheno, get_passed_fields
from scripts.python.pheno.datasets.features import get_column_name, get_status_dict_default
from scripts.python.preprocessing.serialization.routines.pheno_betas_checking import get_pheno_betas_with_common_subjects
from scripts.python.routines.betas import betas_drop_na
from scripts.python.routines.mvals import logit2
from scripts.python.meta.tasks.GPL13534_Blood.routines import perform_test_for_controls
from tqdm import tqdm
import pathlib
import plotly.express as px
import plotly.io as pio
pio.kaleido.scope.mathjax = None


disease = "Schizophrenia"


platform = 'GPL13534'
manifest = get_manifest(platform)

path = f"E:/YandexDisk/Work/pydnameth/draft/03_somewhere/SupplementaryTable2"
base = pd.read_excel(f"{path}/{disease}/base.xlsx", index_col="CpG")
base['importance'] = base['importance'] / base['importance'].sum()

# sources = {
#     'Chuang2017': "Chuang et. al., 2017",
#     'Henderson2019': "Henderson-Smith et. al., 2019",
#     'Kaut2017': "Kaut et. al., 2017"
# }

sources = {
    'Hannon2021': "Hannon et. al., 2021",
    'Walton2015': "Walton et. al., 2015",
}

genes_all = set()
for cpg_id, (cpg, row) in enumerate(base.iterrows()):
    base.at[cpg, 'Gene'] = manifest.at[cpg, 'Gene']
    base.at[cpg, 'CHR'] = manifest.at[cpg, 'CHR']
    base.at[cpg, 'Relation_to_Island'] = manifest.at[cpg, 'Relation_to_Island']
    base.at[cpg, 'UCSC_RefGene_Group'] = manifest.at[cpg, 'UCSC_RefGene_Group']
    genes_raw = manifest.at[cpg, 'Gene']
    genes = genes_raw.split(';')
    genes_all.update(set(genes))
genes_all.remove('non-genic')
genes_all = list(genes_all)
genes_df = pd.DataFrame({'gene':genes_all})
genes_df.to_excel(f"{path}/{disease}/genes.xlsx", index=False)

for s in sources:
    tmp = pd.read_excel(f"{path}/{disease}/{s}.xlsx", index_col="CpG")
    yes_cpgs = list(set(tmp.index.values).intersection(set(base.index.values)))
    no_cpgs = (set(base.index.values) - set(yes_cpgs))
    base.loc[yes_cpgs, sources[s]] = "Yes"
    base.loc[no_cpgs, sources[s]] = "No"

base.to_excel(f"{path}/{disease}/result.xlsx", index_label="CpG")
