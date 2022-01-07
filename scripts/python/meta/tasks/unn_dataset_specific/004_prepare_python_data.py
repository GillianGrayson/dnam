import pandas as pd
import plotly.graph_objects as go
from scripts.python.preprocessing.serialization.routines.filter import get_forbidden_cpgs, manifest_filter
from scripts.python.preprocessing.serialization.routines.pheno_betas_checking import get_pheno_betas_with_common_subjects
from scripts.python.preprocessing.serialization.routines.save import save_pheno_betas_to_pkl
from scripts.python.routines.manifest import get_manifest
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.layout import add_layout
from pathlib import Path


path = f"E:/YandexDisk/Work/pydnameth/datasets/meta/tasks/unn_dataset_specific"
platform = 'GPL21145'
forbidden_types = ["NoCG", "SNP", "MultiHit", "XY"]
gse_dataset = 'GSE164056'

manifest = get_manifest(platform)

path_save = f"{path}/004_prepare_python_data"
Path(f"{path_save}/figs").mkdir(parents=True, exist_ok=True)

fn = f"{path}/002_prepare_pd_for_ChAMP/{gse_dataset}/pheno.xlsx"
df = pd.read_excel(fn)
pheno = df.set_index('Sample_Name')
pheno.index.name = "subject_id"

ys_male = [
    pheno.loc[(pheno['Sex'] == 'F') & (pheno['Sample_Group'] == 'RU'), :].shape[0],
    pheno.loc[(pheno['Sex'] == 'F') & (pheno['Sample_Group'] == 'EU'), :].shape[0]
]
ys_female = [
    pheno.loc[(pheno['Sex'] == 'M') & (pheno['Sample_Group'] == 'RU'), :].shape[0],
    pheno.loc[(pheno['Sex'] == 'M') & (pheno['Sample_Group'] == 'EU'), :].shape[0]
]
fig = go.Figure(data=[
    go.Bar(
        x=['RU', 'EU'],
        y=ys_male,
        name='Female',
        text=ys_male,
        textposition='auto',
        showlegend=True,
    ),
    go.Bar(
        x=['RU', 'EU'],
        y=ys_female,
        name='Male',
        text=ys_female,
        textposition='auto',
        showlegend=True,
    )
])
add_layout(fig, f"", f"Count", "")
fig.update_layout(barmode='group')
fig.update_layout({'colorway': ['red', 'blue']})
fig.update_layout(legend_font_size=20)
fig.update_layout(margin=go.layout.Margin(
    l=60,
    r=10,
    b=40,
    t=50,
    pad=0
))
save_figure(fig, f"{path_save}/figs/bar_Sex")

fn = f"{path}/003_ChAMP_pipeline/myNorm_FunctionalNormalization.txt"
df = pd.read_csv(fn, delimiter="\t", index_col='CpG')
df.index.name = 'CpG'
betas = df.T
betas.index.name = "subject_id"
betas = manifest_filter(betas, manifest)
forbidden_cpgs = get_forbidden_cpgs(f"E:/YandexDisk/Work/pydnameth/datasets/{platform}/manifest/forbidden_cpgs", forbidden_types)
betas = betas.loc[:, ~betas.columns.isin(forbidden_cpgs)]

pheno, betas = get_pheno_betas_with_common_subjects(pheno, betas)
save_pheno_betas_to_pkl(pheno, betas, f"{path_save}")
