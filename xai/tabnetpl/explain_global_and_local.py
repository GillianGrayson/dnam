import numpy as np
from src.models.dnam.tabnet import TabNetModel
import torch
import os
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
from src.datamodules.datasets.dnam_dataset import DNAmDataset
from torch.utils.data import DataLoader
from scipy.sparse import csc_matrix
from scipy.special import log_softmax
from scipy.special import softmax
import plotly.graph_objects as go
from collections import Counter
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.scatter import add_scatter_trace
from scripts.python.routines.plot.violin import add_violin_trace
from scripts.python.routines.plot.layout import add_layout
from tqdm import tqdm
import pandas as pd
import dotenv
from pytorch_lightning import LightningModule, LightningDataModule
from torch.utils.data import DataLoader, ConcatDataset, Subset
from pytorch_lightning import seed_everything
import hydra
from src.utils import utils
from omegaconf import DictConfig
import shap
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay
import plotly.figure_factory as ff


log = utils.get_logger(__name__)

dotenv.load_dotenv(override=True)

@hydra.main(config_path="../../configs/", config_name="main_xai.yaml")
def main(config: DictConfig):

    statuses = [
        'Schizophrenia',
        'First episode psychosis',
        'Depression',
        'Control',
    ]

    for st in statuses:
        Path(f"{st}").mkdir(parents=True, exist_ok=True)

    num_top_features = config.num_top_features
    num_examples = config.num_examples

    if "seed" in config:
        seed_everything(config.seed)

    checkpoint_path = config.checkpoint_path
    checkpoint_name = config.checkpoint_name
    explainer_type = config.explainer_type

    model = TabNetModel.load_from_checkpoint(checkpoint_path=f"{checkpoint_path}/{checkpoint_name}")
    model.produce_probabilities = True
    model.eval()
    model.freeze()

    # Init Lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule.setup()

    manifest = pd.read_excel(f"{datamodule.path}/manifest.xlsx", index_col='CpG')

    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    test_dataloader = datamodule.test_dataloader()
    dataset = ConcatDataset([train_dataloader.dataset, val_dataloader.dataset, test_dataloader.dataset])
    all_dataloader = DataLoader(
        dataset,
        batch_size=config.datamodule.batch_size,
        num_workers=config.datamodule.num_workers,
        pin_memory=config.datamodule.pin_memory,
        shuffle=True
    )
    common_df = pd.merge(datamodule.pheno, datamodule.betas, left_index=True, right_index=True)

    outs_real_all = np.empty(0, dtype=int)
    outs_pred_all = np.empty(0, dtype=int)
    outs_prob_all = np.empty(shape=(0, len(statuses)), dtype=int)
    for x, outs_real, indexes in tqdm(test_dataloader):
        outs_real = outs_real.cpu().detach().numpy()
        outs_prob = model(x).cpu().detach().numpy()
        outs_pred = np.argmax(outs_prob, axis=1)
        outs_real_all = np.append(outs_real_all, outs_real, axis=0)
        outs_pred_all = np.append(outs_pred_all, outs_pred, axis=0)
        outs_prob_all = np.append(outs_prob_all, outs_prob, axis=0)

    conf_mtx = confusion_matrix(outs_real_all, outs_pred_all)
    fig = ff.create_annotated_heatmap(conf_mtx, x=statuses, y=statuses, colorscale='Viridis')
    fig.add_annotation(dict(font=dict(color="black", size=14),
                            x=0.5,
                            y=-0.1,
                            showarrow=False,
                            text="Predicted value",
                            xref="paper",
                            yref="paper"))
    fig.add_annotation(dict(font=dict(color="black", size=14),
                            x=-0.33,
                            y=0.5,
                            showarrow=False,
                            text="Real value",
                            textangle=-90,
                            xref="paper",
                            yref="paper"))
    fig.update_layout(margin=dict(t=50, l=200))
    fig['data'][0]['showscale'] = True
    save_figure(fig, 'test_confusion_matrix')
    roc_auc = roc_auc_score(outs_real_all, outs_prob_all, average='macro', multi_class='ovr')
    log.info(f"roc_auc for test set: {roc_auc}")

    feature_importances = np.zeros((model.hparams.input_dim))
    for data, targets, indexes in tqdm(train_dataloader):
        M_explain, masks = model.forward_masks(data)
        feature_importances += M_explain.sum(dim=0).cpu().detach().numpy()

    feature_importances = feature_importances / np.sum(feature_importances)
    feature_importances_df = pd.DataFrame.from_dict(
        {
            'feature': datamodule.betas.columns.values,
            'importance': feature_importances
        }
    )
    feature_importances_df.set_index('feature', inplace=True)
    feature_importances_df.to_excel("./feature_importances.xlsx", index=True)

    background_dataloader = test_dataloader

    background, outs_real, indexes = next(iter(background_dataloader))
    probs = model(background)
    background_np = background.cpu().detach().numpy()
    outs_real_np = outs_real.cpu().detach().numpy()
    indexes_np = indexes.cpu().detach().numpy()
    probs_np = probs.cpu().detach().numpy()
    outs_pred_np = np.argmax(probs_np, axis=1)

    is_correct_pred = (outs_real_np == outs_pred_np)
    mistakes_ids = np.where(is_correct_pred == False)[0]

    if explainer_type == "deep":
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(background)
    elif explainer_type == "kernel":
        def proba(X):
            X = torch.from_numpy(X)
            tmp = model(X)
            return tmp.cpu().detach().numpy()

        explainer = shap.KernelExplainer(proba, background_np)
        shap_values = explainer.shap_values(background_np)
    else:
        raise ValueError("Unsupported explainer type")

    shap.summary_plot(
        shap_values=shap_values,
        features=background_np,
        feature_names=datamodule.betas.columns.values,
        max_display=num_top_features,
        class_names=statuses,
        class_inds=list(range(len(statuses))),
        plot_size=(14, 10),
        show=False,
        color=plt.get_cmap("Set1")
    )
    plt.savefig('SHAP_bar.png')
    plt.savefig('SHAP_bar.pdf')
    plt.close()

    for st_id, st in enumerate(statuses):
        shap.summary_plot(
            shap_values=shap_values[st_id],
            features=background_np,
            feature_names=datamodule.betas.columns.values,
            max_display=num_top_features,
            plot_size=(14, 10),
            plot_type="violin",
            title=st,
            show=False
        )
        plt.savefig(f"{st}/beeswarm.png")
        plt.savefig(f"{st}/beeswarm.pdf")
        plt.close()

    print(f"Number of errors: {len(mistakes_ids)}")
    for m_id in mistakes_ids:
        subj_cl = outs_real_np[m_id]
        subj_pred_cl = outs_pred_np[m_id]
        subj_global_id = indexes_np[m_id]
        subj_name = datamodule.betas.index.values[subj_global_id]
        for st_id, st in enumerate(statuses):
            probs_real = probs_np[m_id, st_id]
            probs_expl = explainer.expected_value[st_id] + sum(shap_values[st_id][m_id])
            if abs(probs_real - probs_expl) > 1e-6:
                print(f"diff between prediction: {abs(probs_real - probs_expl)}")
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[st_id][m_id],
                    base_values=explainer.expected_value[st_id],
                    data=background_np[m_id],
                    feature_names=datamodule.betas.columns.values
                ),
                max_display=num_top_features,
                show=False
            )
            fig = plt.gcf()
            fig.set_size_inches(18, 10, forward=True)
            Path(f"errors/real({statuses[subj_cl]})_pred({statuses[subj_pred_cl]})/{subj_name}").mkdir(parents=True, exist_ok=True)
            fig.savefig(f"errors/real({statuses[subj_cl]})_pred({statuses[subj_pred_cl]})/{subj_name}/waterfall_{st}.pdf")
            fig.savefig(f"errors/real({statuses[subj_cl]})_pred({statuses[subj_pred_cl]})/{subj_name}/waterfall_{st}.png")
            plt.close()

    passed_examples = {x: 0 for x in range(len(statuses))}
    for subj_id in range(background_np.shape[0]):
        subj_cl = outs_real_np[subj_id]
        if passed_examples[subj_cl] < num_examples:
            subj_global_id = indexes_np[subj_id]
            subj_name = datamodule.betas.index.values[subj_global_id]

            for st_id, st in enumerate(statuses):

                probs_real = probs_np[subj_id, st_id]
                probs_expl = explainer.expected_value[st_id] + sum(shap_values[st_id][subj_id])
                if abs(probs_real - probs_expl) > 1e-6:
                    print(f"diff between prediction: {abs(probs_real - probs_expl)}")

                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_values[st_id][subj_id],
                        base_values=explainer.expected_value[st_id],
                        data=background_np[subj_id],
                        feature_names=datamodule.betas.columns.values
                    ),
                    max_display=num_top_features,
                    show=False
                )
                fig = plt.gcf()
                fig.set_size_inches(18, 10, forward=True)
                Path(f"{statuses[subj_cl]}/{passed_examples[subj_cl]}_{subj_name}").mkdir(parents=True, exist_ok=True)
                fig.savefig(f"{statuses[subj_cl]}/{passed_examples[subj_cl]}_{subj_name}/waterfall_{st}.pdf")
                fig.savefig(f"{statuses[subj_cl]}/{passed_examples[subj_cl]}_{subj_name}/waterfall_{st}.png")
                plt.close()
            passed_examples[subj_cl] += 1

    mean_abs_shap_vals = np.sum([np.mean(np.absolute(shap_values[cl_id]), axis=0) for cl_id, cl in enumerate(statuses)], axis=0)
    order = np.argsort(mean_abs_shap_vals)[::-1]
    features = datamodule.betas.columns.values[order]
    features_best = features[0:num_top_features]

    for feat_id, feat in enumerate(features_best):
        fig = go.Figure()
        for st_id, st in enumerate(statuses):
            class_shap_values = shap_values[st_id][:, order[feat_id]]
            real_values = background_np[:, order[feat_id]]
            add_scatter_trace(fig, real_values, class_shap_values, st)
        add_layout(fig, f"{feat} ({manifest.at[feat, 'Gene']})", f"SHAP values", f"")
        fig.update_layout({'colorway': px.colors.qualitative.Set1})
        Path(f"features/scatter").mkdir(parents=True, exist_ok=True)
        save_figure(fig, f"features/scatter/{feat_id}_{feat}")

        fig = go.Figure()
        for st_id, st in enumerate(statuses):
            add_violin_trace(fig, common_df.loc[common_df['Status'] == st_id, feat].values, st)
        add_layout(fig, "", f"{feat} ({manifest.at[feat, 'Gene']})", f"")
        fig.update_xaxes(showticklabels=False)
        fig.update_layout({'colorway': px.colors.qualitative.Set1})
        Path(f"features/violin").mkdir(parents=True, exist_ok=True)
        save_figure(fig, f"features/violin/{feat_id}_{feat}")

    for st_id, st in enumerate(statuses):
        class_shap_values = shap_values[st_id]
        shap_abs = np.absolute(class_shap_values)
        shap_mean_abs = np.mean(shap_abs, axis=0)
        order = np.argsort(shap_mean_abs)[::-1]
        features = datamodule.betas.columns.values
        features_best = features[order[0:num_top_features]]
        subject_indices = indexes.flatten().cpu().detach().numpy()
        subjects = datamodule.betas.index.values[subject_indices]
        d = {'subjects': subjects}
        for f_id in range(0, num_top_features):
            feat = features_best[f_id]
            curr_beta = background_np[:, order[f_id]]
            curr_shap = class_shap_values[:, order[f_id]]
            d[f"{feat}_beta"] = curr_beta
            d[f"{feat}_shap"] = curr_shap
        df_features = pd.DataFrame(d)
        df_features.to_excel(f"{st}/shap.xlsx", index=False)


if __name__ == "__main__":
    main()
