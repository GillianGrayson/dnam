import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scripts.python.pheno.datasets.filter import filter_pheno
from scripts.python.pheno.datasets.features import get_column_name, get_status_dict, get_sex_dict
from scripts.python.routines.plot.scatter import add_scatter_trace
import plotly.graph_objects as go
import pathlib
from scripts.python.routines.manifest import get_manifest
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.layout import add_layout
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from scipy.stats import chi2
from pyod.models.abod import ABOD
from pyod.models.knn import KNN
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.sos import SOS
from pyod.models.suod import SUOD


def mahalanobis(x=None, data=None, cov=None):
    """Compute the Mahalanobis Distance between each row of x and the data
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    x_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = np.linalg.inv(cov)
    left = np.dot(x_mu, inv_covmat)
    mahal = np.dot(left, x_mu.T)
    return mahal.diagonal()


def calc_metrics(model, X, y, comment, params):
    y_pred = model.predict(X)
    score = model.score(X, y)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    params[f'{comment} R2'] = score
    params[f'{comment} RMSE'] = rmse
    params[f'{comment} MAE'] = mae
    return y_pred

def analyze_outliers(ctrl, features, data_type):

    corr_df = pd.DataFrame(data=np.zeros((len(features), 2)), index=features, columns=['pearson_corr', 'pearson_pval'])
    for f in features:
        corr, pval = stats.pearsonr(ctrl.loc[:, f].values, ctrl.loc[:, 'Age'].values)
        corr_df.at[f, 'pearson_corr'] = corr
        corr_df.at[f, 'pearson_pval'] = pval
    _, corr_df['pearson_pval_fdr_bh'], _, _ = multipletests(corr_df.loc[:, 'pearson_pval'].values, 0.05, method='fdr_bh')
    aa_features = corr_df.index[corr_df['pearson_pval_fdr_bh'] < 0.01].values

    features = aa_features

    iqr_feature = []
    for f in features:
        q1 = ctrl[f].quantile(0.25)
        q3 = ctrl[f].quantile(0.75)
        iqr = q3 - q1
        filter = (ctrl[f] >= q1 - 1.5 * iqr) & (ctrl[f] <= q3 + 1.5 * iqr)
        iqr_feature.append(f"{f}_IsIQR")
        ctrl[f"{f}_IsIQR"] = filter
    ctrl[f"NumIQR_{data_type}"] = len(aa_features) - ctrl[iqr_feature].sum(axis=1)
    ctrl[f"PassedByNumIQR_{data_type}"] = ctrl[f"NumIQR_{data_type}"] < 1

    X_ctrl = ctrl.loc[:, features].to_numpy()

    lof = LocalOutlierFactor()
    ctrl[f"LocalOutlierFactor_{data_type}"] = lof.fit_predict(X_ctrl)
    ctrl[f"LocalOutlierFactor_{data_type}"].replace({1: True, -1: False}, inplace=True)
    iso = IsolationForest()
    ctrl[f"IsolationForest_{data_type}"] = iso.fit_predict(X_ctrl)
    ctrl[f"IsolationForest_{data_type}"].replace({1: True, -1: False}, inplace=True)
    ee = OneClassSVM()
    ctrl[f"OneClassSVM_{data_type}"] = ee.fit_predict(X_ctrl)
    ctrl[f"OneClassSVM_{data_type}"].replace({1: True, -1: False}, inplace=True)
    ctrl[f'mahalanobis_d_{data_type}'] = mahalanobis(x=ctrl[features], data=ctrl[features])
    ctrl[f'mahalanobis_p_{data_type}'] = 1 - chi2.cdf(ctrl[f'mahalanobis_d_{data_type}'], 3)
    ctrl[f"PassedByMahalanobis_{data_type}"] = ctrl[f'mahalanobis_p_{data_type}'] <= 0.05

    outlier_fraction = 0.4
    classifiers = {
        'ABOD': ABOD(contamination=outlier_fraction),
        'KNN': KNN(contamination=outlier_fraction),
        'COPOD': COPOD(contamination=outlier_fraction),
        'ECOD': ECOD(contamination=outlier_fraction),
        'SOS': SOS(contamination=outlier_fraction),
        'SUOD': SUOD(contamination=outlier_fraction)
    }
    outlier_types = ["PassedByImmunoAgeDiff", f"PassedByNumIQR_{data_type}"]
    for i, (clf_name, clf) in enumerate(classifiers.items()):
        clf.fit(X_ctrl)
        ctrl[f"{clf_name}_{data_type}"] = clf.predict(X_ctrl)
        ctrl[f"{clf_name}_{data_type}"].replace({0: True, 1: False}, inplace=True)
        outlier_types.append(f"{clf_name}_{data_type}")

    outlier_types += [f"{x}_{data_type}" for x in ["LocalOutlierFactor", "IsolationForest", "OneClassSVM", "PassedByMahalanobis"]]


    ctrl[f"OutInPartOfAAFeatures_{data_type}"] = np.sum((np.abs(stats.zscore(ctrl[aa_features])) > 3), axis=1) / len(aa_features)
    ctrl[f"PassedByOutInPartOfAAFeatures_{data_type}"] =  ctrl[f"OutInPartOfAAFeatures_{data_type}"] < 0.01

    outlier_types += [f"PassedByOutInPartOfAAFeatures_{data_type}"]

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(ctrl[features])
    for pc_id in range(pcs.shape[1]):
        ctrl[f"PC_{pc_id + 1}_{data_type}"] = pcs[:, pc_id]
        ctrl[f"PC_{pc_id + 1}_{data_type}_log"] = np.sign(pcs[:, pc_id]) * np.log10(1 + np.abs(pcs[:, pc_id]))

    pc_plot_list = [
        {'x_col': f"PC_1_{data_type}", 'y_col': f"PC_2_{data_type}", 'x_name': f"PC1", 'y_name': f"PC2",
         'path': f"{path_save}/outliers/{data_type}", 'name': '0_PC'},
        {'x_col': f"PC_1_{data_type}_log", 'y_col': f"PC_2_{data_type}_log", 'x_name': f"sign(PC1) log(1 + |PC1|)",
         'y_name': f"sign(PC2) log(1 + |PC2|)", 'path': f"{path_save}/outliers/{data_type}", 'name': '0_PC_log'}
    ]
    for pc_plot in pc_plot_list:
        fig = go.Figure()
        add_scatter_trace(fig, ctrl.loc[ctrl['Source'] == 1, pc_plot['x_col']].values, ctrl.loc[ctrl['Source'] == 1, pc_plot['y_col']].values, f"First and Second")
        add_scatter_trace(fig, ctrl.loc[ctrl['Source'] == 2, pc_plot['x_col']].values, ctrl.loc[ctrl['Source'] == 2, pc_plot['y_col']].values, f"Third and Fourth")
        add_layout(fig, pc_plot['x_name'], pc_plot['y_name'], f"")
        fig.update_layout({'colorway': ['red', 'blue']})
        fig.update_layout(legend_font_size=20)
        fig.update_layout(
            margin=go.layout.Margin(
                l=110,
                r=20,
                b=75,
                t=45,
                pad=0
            )
        )
        save_figure(fig, f"{pc_plot['path']}/{pc_plot['name']}")

    for ot_id, ot in enumerate(outlier_types):
        n_total = ctrl.loc[(ctrl[ot] == True), :].shape[0]
        n_in_intersection = ctrl.loc[(ctrl[ot] == True) & (ctrl['PassedByImmunoAgeDiff'] == True),:].shape[0]
        print(f"Number of common subject of {ot} with PassedByImmunoAgeDiff: {n_in_intersection} from {n_total}")
        pc_plot_list[0]['name'] = f"{ot_id + 1}_{ot}"
        pc_plot_list[1]['name'] = f"{ot_id + 1}_{ot}_log"
        for pc_plot in pc_plot_list:
            fig = go.Figure()
            add_scatter_trace(fig, ctrl.loc[ctrl[ot] == True, pc_plot['x_col']].values, ctrl.loc[ctrl[ot] == True, pc_plot['y_col']].values, f"Inlier")
            add_scatter_trace(fig, ctrl.loc[ctrl[ot] == False, pc_plot['x_col']].values, ctrl.loc[ctrl[ot] == False, pc_plot['y_col']].values, f"Outlier")
            add_layout(fig, pc_plot['x_name'], pc_plot['y_name'], f"{ot}")
            fig.update_layout({'colorway': ['blue', 'red']})
            fig.update_layout(legend_font_size=20)
            fig.update_layout(
                margin=go.layout.Margin(
                    l=110,
                    r=20,
                    b=75,
                    t=85,
                    pad=0
                )
            )
            save_figure(fig, f"{pc_plot['path']}/{pc_plot['name']}")

thld_abs_diff = 16

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
pheno = pd.read_pickle(f"{path}/{platform}/{dataset}/pheno_xtd.pkl")
pheno = filter_pheno(dataset, pheno, continuous_vars, categorical_vars)
pheno['Source'] = 1

part_3_4 = pd.read_excel(f"{path}/{platform}/{dataset}/data/immuno/part3_part4_with_age_sex.xlsx", index_col='ID')
part_3_4 = part_3_4[~part_3_4.index.str.startswith(('Q', 'H'))]
part_3_4['Group'] = 'Control'
part_3_4['Source'] = 2

pheno.set_index('ID', inplace=True)
df = pheno.append(part_3_4, verify_integrity=True)
df[f'ImmunoAgeDiff'] = df[f'ImmunoAge'] - df[f'Age']
df[f"PassedByImmunoAgeDiff"] = abs(df[f'ImmunoAgeDiff']) <= thld_abs_diff

ctrl = df.loc[df['Group'] == 'Control']
esrd = df.loc[df['Group'] == 'ESRD']

path_save = f"{path}/{platform}/{dataset}/special/017_outlier_detection_in_controls/"
pathlib.Path(f"{path_save}/outliers/origin").mkdir(parents=True, exist_ok=True)
pathlib.Path(f"{path_save}/outliers/scaled").mkdir(parents=True, exist_ok=True)

with open(f'{path}/{platform}/{dataset}/features/immuno.txt') as f:
    features = f.read().splitlines()

scalers = {}
features_scaled = []
for f in features:
    scaler = StandardScaler()
    scaler.fit(ctrl.loc[:, f].values.reshape(-1, 1))
    scalers[f] = scaler
    features_scaled.append(f"{f}_scaled")
    ctrl[f"{f}_scaled"] = scaler.transform(ctrl.loc[:, f].values.reshape(-1, 1))

analyze_outliers(ctrl, features, 'origin')
analyze_outliers(ctrl, features_scaled, 'scaled')

ctrl.to_excel(f'{path_save}/ctrl.xlsx', index=True)
