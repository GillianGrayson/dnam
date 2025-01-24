import pandas as pd
import numpy as np
from scripts.python.routines.manifest import get_manifest
from scripts.python.routines.sections import get_sections
import re
import upsetplot as upset
from upsetplot import UpSet
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.scatter import add_scatter_trace
from scripts.python.routines.plot.layout import add_layout
from scripts.python.routines.manifest import get_genes_list
from pathlib import Path
import scripts.python.routines.plot.venn as venn


def plot_upset(genes_universe, dict_of_lists, path_save, suffix):
    upset_df = pd.DataFrame(index=list(genes_universe))
    for k, v in dict_of_lists.items():
        upset_df[k] = upset_df.index.isin(v)
    upset_df = upset_df.set_index(list(dict_of_lists.keys()))
    fig = upset.UpSet(upset_df, subset_size='count', show_counts=True, min_degree=1, sort_categories_by=None).plot()
    plt.savefig(f"{path_save}/figs/upset_{suffix}.png", bbox_inches='tight')
    plt.savefig(f"{path_save}/figs/upset_{suffix}.pdf", bbox_inches='tight')


path = f"E:/YandexDisk/Work/pydnameth/datasets"
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')

folder_name = f"proteomics"
path_save = f"{path}/meta/tasks/{folder_name}"

tissues = ['Brain', 'Liver', 'Blood']

platform = 'GPL13534'
manifest = get_manifest(platform)

age_corr = 'spearman' # 'pearson'
corr_type = 'fdr_bh' # 'bonferroni'
thld_age = 0.01
thld_sex = 0.01

SS_lists = {}
AA_lists = {}
SSAA_lists = {}

proteomic_path = f"E:/YandexDisk/Work/pydnameth/methylation_and_proteomic"
t1 = pd.read_excel(f"{proteomic_path}/proteomic_data/T1.xlsx", index_col='ID')
t4 = pd.read_excel(f"{proteomic_path}/proteomic_data/T4.xlsx", index_col='ID')
prot = pd.merge(t1, t4, left_index=True, right_index=True)
SS_prot = prot.loc[prot['q.Sex'] < 0.05, :]
SS_prot_genes = get_genes_list(SS_prot, 'EntrezGeneSymbol', [np.nan], r'[.;]+')
SS_lists['Proteomic'] = SS_prot_genes
print(f"Proteomic SS genes: {len(SS_prot_genes)}")
AA_prot = prot.loc[prot['q.Age'] < 0.05, :]
AA_prot_genes = get_genes_list(AA_prot, 'EntrezGeneSymbol', [np.nan], r'[.;]+')
AA_lists['Proteomic'] = AA_prot_genes
print(f"Proteomic AA genes: {len(AA_prot_genes)}")
SSAA_prot = prot.loc[(prot['q.Sex'] < 0.05) & (prot['q.Age'] < 0.05), :]
SSAA_prot_genes = get_genes_list(SSAA_prot, 'EntrezGeneSymbol', [np.nan], r'[.;]+')
SSAA_lists['Proteomic'] = SSAA_prot_genes
print(f"Proteomic SS genes: {len(SSAA_prot_genes)}")

genes_universe = set(get_genes_list(prot, 'EntrezGeneSymbol', [np.nan], r'[.;]+'))

for tissue in tissues:
    tmp_path = f"{path_save}/{tissue}"

    metrics = ['pearson_r', 'pearson_pval', 'spearman_r', 'spearman_pval', 'mannwhitney_stat', 'mannwhitney_pval']
    corr_types = ['fdr_bh', 'bonferroni']
    stats = pd.read_pickle(f"{tmp_path}/stats.pkl")
    genes_universe.update(set(get_genes_list(stats, 'Gene', ['non-genic'], ';+')))

    AA = stats.loc[(stats[f"{age_corr}_pval_{corr_type}"] < thld_age), :]
    #AA = stats.loc[(stats[f"spearman_r"] > 0.5) | (stats[f"spearman_r"] < -0.5), :]
    AA_genes = get_genes_list(AA, 'Gene', ['non-genic'], ';+')
    AA_lists[tissue] = AA_genes
    print(f"{tissue} AA genes: {len(AA_genes)}")

    SS = stats.loc[(stats[f"mannwhitney_pval_{corr_type}"] < thld_sex), :]
    SS_genes = get_genes_list(SS, 'Gene', ['non-genic'], ';+')
    SS_lists[tissue] = SS_genes
    print(f"{tissue} SS genes: {len(SS_genes)}")

    print(f"{tissue} SSAA checking: {len(set(AA_genes).intersection(set(SS_genes)))}")

    #SSAA = stats.loc[(stats[f"mannwhitney_pval_{corr_type}"] < thld_sex) & (stats[f"{age_corr}_pval_{corr_type}"] < thld_age), :]
    #SSAA_genes = get_genes_list(SSAA, 'Gene', ['non-genic'])
    SSAA_genes = list(set(AA_genes).intersection(set(SS_genes)))
    SSAA_lists[tissue] = SSAA_genes
    print(f"{tissue} SSAA genes: {len(SSAA_genes)}")

common_genes = {}

AA_sets = [set(x) for x in AA_lists.values()]
AA_tags = [x for x in AA_lists.keys()]
AA_sections = get_sections(AA_sets)
plot_upset(genes_universe, AA_lists, path_save, 'AA')
common_genes['AA'] = list(AA_sections['1'*len(AA_tags)])
labels = venn.get_labels(list(AA_lists.values()), fill=['number'])
fig, ax = venn.venn4(labels, names=list(AA_lists.keys()))
plt.savefig(f"{path_save}/figs/venn_AA.png", bbox_inches='tight')
plt.savefig(f"{path_save}/figs/venn_AA.pdf", bbox_inches='tight')
plt.close('all')

SS_sets = [set(x) for x in SS_lists.values()]
SS_tags = [x for x in SS_lists.keys()]
SS_sections = get_sections(SS_sets)
plot_upset(genes_universe, SS_lists, path_save, 'SS')
common_genes['SS'] = list(SS_sections['1'*len(SS_tags)])
labels = venn.get_labels(list(SS_lists.values()), fill=['number'])
fig, ax = venn.venn4(labels, names=list(SS_lists.keys()))
plt.savefig(f"{path_save}/figs/venn_SS.png", bbox_inches='tight')
plt.savefig(f"{path_save}/figs/venn_SS.pdf", bbox_inches='tight')
plt.close('all')

SSAA_sets = [set(x) for x in SSAA_lists.values()]
SSAA_tags = [x for x in SSAA_lists.keys()]
SSAA_sections = get_sections(SSAA_sets)
plot_upset(genes_universe, SSAA_lists, path_save, 'SSAA')
common_genes['SSAA'] = list(SSAA_sections['1'*len(SSAA_tags)])
labels = venn.get_labels(list(SSAA_lists.values()), fill=['number'])
fig, ax = venn.venn4(labels, names=list(SSAA_lists.keys()))
plt.savefig(f"{path_save}/figs/venn_SSAA.png", bbox_inches='tight')
plt.savefig(f"{path_save}/figs/venn_SSAA.pdf", bbox_inches='tight')
plt.close('all')

np.savetxt(f"{path_save}/genes_SS.txt", common_genes['SS'], fmt="%s")
np.savetxt(f"{path_save}/genes_AA.txt", common_genes['AA'], fmt="%s")
np.savetxt(f"{path_save}/genes_SSAA.txt", common_genes['SSAA'], fmt="%s")

upset_df = pd.DataFrame(index=list(genes_universe))
for k, v in common_genes.items():
    upset_df[k] = upset_df.index.isin(v)
upset_df = upset_df.set_index(list(common_genes.keys()))
fig = UpSet(upset_df, subset_size='count', show_counts=True, min_degree=1, sort_categories_by=None)
fig.style_subsets(present="AA", absent="SS", facecolor='b')
fig.style_subsets(present="SS", absent="AA", facecolor='g')
fig.style_subsets(present=["SS", "AA"], facecolor='r')
fig.plot()
plt.savefig(f"{path_save}/figs/upset_common.png", bbox_inches='tight')
plt.savefig(f"{path_save}/figs/upset_common.pdf", bbox_inches='tight')

common_sets = [set(x) for x in common_genes.values()]
common_tags = [x for x in common_genes.keys()]
common_sections = get_sections(common_sets)

if Path(f"{proteomic_path}/GTEx.pkl").is_file():
    GTEx = pd.read_pickle(f"{proteomic_path}/GTEx.pkl")
else:
    GTEx = pd.read_excel(f"{proteomic_path}/GTEx.xlsx", index_col='Name')
    GTEx.to_pickle(f"{proteomic_path}/GTEx.pkl")

AA_genes = list(common_sections['100'])
SS_genes = list(common_sections['010'])
SSAA_genes = list(common_sections['111'])

AA_GTEx = GTEx.loc[GTEx['Description'].isin(AA_genes), :]
SS_GTEx = GTEx.loc[GTEx['Description'].isin(SS_genes), :]
SSAA_GTEx = GTEx.loc[GTEx['Description'].isin(SSAA_genes), :]

tissue_names = {
    "Whole Blood": "Whole Blood",
    "Brain - Frontal Cortex BA9": "Brain Frontal Cortex",
    "Liver": "Liver"
}
for x_type,y_type in [["Whole Blood", "Brain - Frontal Cortex BA9"], ["Whole Blood", "Liver"], ["Brain - Frontal Cortex BA9", "Liver"]]:
    fig = go.Figure()
    xs = AA_GTEx.loc[:, x_type].values
    ys = AA_GTEx.loc[:, y_type].values
    fig.add_trace(
        go.Scatter(
            x=np.log10(AA_GTEx.loc[:, x_type].values + 1.0),
            y=np.log10(AA_GTEx.loc[:, y_type].values + 1.0),
            showlegend=True,
            name="Age-Associated (AA)",
            mode="markers",
            marker=dict(
                size=3,
                opacity=0.25,
                line=dict(
                    width=0.1
                )
            )
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.log10(SS_GTEx.loc[:, x_type].values + 1.0),
            y=np.log10(SS_GTEx.loc[:, y_type].values + 1.0),
            showlegend=True,
            name="Sex-Specific (SS)",
            mode="markers",
            marker=dict(
                size=8,
                opacity=0.8,
                line=dict(
                    width=1
                )
            )
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.log10(SSAA_GTEx.loc[:, x_type].values + 1.0),
            y=np.log10(SSAA_GTEx.loc[:, y_type].values + 1.0),
            showlegend=True,
            name="Sex-Specific Age-Associated (SSAA)",
            mode="markers+text",
            text=SSAA_GTEx.loc[:, 'Description'].values,
            textposition="middle right",
            marker=dict(
                size=10,
                opacity=0.8,
                line=dict(
                    width=1
                )
            )
        )
    )
    add_layout(fig, f"log10(TPM + 1)", f"log10(TPM + 1)", f"")
    fig.update_layout({'colorway': ["blue", "green", "red"]})
    fig.update_layout(
        margin=go.layout.Margin(
        l=80,
        r=20,
        b=80,
        t=50,
        pad=0
        )
    )
    save_figure(fig, f"{path_save}/figs/x({tissue_names[x_type]})_y({tissue_names[y_type]})")
