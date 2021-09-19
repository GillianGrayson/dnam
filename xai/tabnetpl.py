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


log = utils.get_logger(__name__)

dotenv.load_dotenv(override=True)

@hydra.main(config_path="../configs/", config_name="main_xai.yaml")
def main(config: DictConfig):

    # class_names = [
    #     "Control",
    #     "Schizophrenia",
    #     "Depression",
    #     "Parkinson"
    # ]

    # class_names = [
    #     "Schizophrenia Control",
    #     "Schizophrenia",
    #     "Depression Control",
    #     "Depression",
    #     "Parkinson Control",
    #     "Parkinson"
    # ]

    class_names = [
        "Schizophrenia",
        "Depression",
        "Parkinson"
    ]

    for cl in class_names:
        Path(f"{cl}").mkdir(parents=True, exist_ok=True)

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

    for name, ids in {"all": datamodule.ids_all,
                      "train_val": datamodule.ids_train_val,
                      "train": datamodule.ids_train,
                      "val": datamodule.ids_val,
                      "test": datamodule.ids_test}.items():
        status_counts = pd.DataFrame(Counter(datamodule.dataset.ys[ids]), index=[0])
        status_counts.rename({x_id: x for x_id, x in enumerate(class_names)}, inplace=True)
        status_counts = pd.DataFrame(Counter([class_names[x] for x in datamodule.dataset.ys[ids]]), index=[0])
        status_counts = status_counts.reindex(sorted(status_counts.columns), axis=1)
        plot = status_counts.plot.bar(color={x:plt.get_cmap("Set1").colors[x_id] for x_id, x in enumerate(class_names)})
        plt.xlabel("Status", fontsize=15)
        plt.ylabel("Count", fontsize=15)
        plt.xticks([])
        plt.axis('auto')
        fig = plot.get_figure()
        fig.savefig(f"bar_{name}.pdf")
        fig.savefig(f"bar_{name}.png")
        plt.close()

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
    outs_prob_all = np.empty(shape=(0, len(class_names)), dtype=int)
    for x, outs_real, indexes in tqdm(test_dataloader):
        outs_real = outs_real.cpu().detach().numpy()
        outs_prob = model(x).cpu().detach().numpy()
        outs_pred = np.argmax(outs_prob, axis=1)
        outs_real_all = np.append(outs_real_all, outs_real, axis=0)
        outs_pred_all = np.append(outs_pred_all, outs_pred, axis=0)
        outs_prob_all = np.append(outs_prob_all, outs_prob, axis=0)

    conf_mtx = confusion_matrix(outs_real_all, outs_pred_all)
    disp = ConfusionMatrixDisplay(conf_mtx, display_labels=[x.replace(' ', '\n') for x in class_names])
    disp.plot()
    fig = plt.gcf()
    fig.set_size_inches(8, 6, forward=True)
    plt.savefig('test_confusion_matrix.png')
    plt.savefig('test_confusion_matrix.pdf')
    plt.close()

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

    background_dataloader = train_dataloader

    background, outs_real, indexes = next(iter(background_dataloader))
    probs = model(background)
    background_np = background.cpu().detach().numpy()
    outs_real_np = outs_real.cpu().detach().numpy()
    indexes_np = indexes.cpu().detach().numpy()
    probs_np = probs.cpu().detach().numpy()

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
        class_names=class_names,
        class_inds=list(range(len(class_names))),
        plot_size=(12, 8),
        show=False,
        color=plt.get_cmap("Set1")
    )
    plt.savefig('SHAP_bar.png')
    plt.savefig('SHAP_bar.pdf')
    plt.close()

    for cl_id, cl in enumerate(class_names):
        shap.summary_plot(
            shap_values=shap_values[cl_id],
            features=background_np,
            feature_names=datamodule.betas.columns.values,
            max_display=num_top_features,
            plot_size=(12, 8),
            plot_type="violin",
            title=cl,
            show=False
        )
        plt.savefig(f"{cl}/beeswarm.png")
        plt.savefig(f"{cl}/beeswarm.pdf")
        plt.close()

    passed_examples = {x: 0 for x in range(len(class_names))}
    for subj_id in range(background_np.shape[0]):
        subj_cl = outs_real_np[subj_id]
        if passed_examples[subj_cl] < num_examples:
            subj_global_id = indexes_np[subj_id]
            subj_name = datamodule.betas.index.values[subj_global_id]

            for cl_id, cl in enumerate(class_names):

                probs_real = probs_np[subj_id, cl_id]
                probs_expl = explainer.expected_value[cl_id] + sum(shap_values[cl_id][subj_id])
                if abs(probs_real - probs_expl) > 1e-8:
                    raise ValueError("Model's probability differs explanation's probability")

                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_values[cl_id][subj_id],
                        base_values=explainer.expected_value[cl_id],
                        data=background_np[subj_id],
                        feature_names=datamodule.betas.columns.values
                    ),
                    max_display=num_top_features,
                    show=False
                )
                fig = plt.gcf()
                fig.set_size_inches(18, 8, forward=True)
                Path(f"{class_names[subj_cl]}/{passed_examples[subj_cl]}_{subj_name}").mkdir(parents=True, exist_ok=True)
                fig.savefig(f"{class_names[subj_cl]}/{passed_examples[subj_cl]}_{subj_name}/waterfall_{cl}.pdf")
                fig.savefig(f"{class_names[subj_cl]}/{passed_examples[subj_cl]}_{subj_name}/waterfall_{cl}.png")
                plt.close()
            passed_examples[subj_cl] += 1

    mean_abs_shap_vals = np.sum([np.mean(np.absolute(shap_values[cl_id]), axis=0) for cl_id, cl in enumerate(class_names)], axis=0)
    order = np.argsort(mean_abs_shap_vals)[::-1]
    features = datamodule.betas.columns.values[order]
    features_best = features[0:num_top_features]


    for feat_id, feat in enumerate(features_best):

        fig = go.Figure()
        for cl_id, cl in enumerate(class_names):
            class_shap_values = shap_values[cl_id][:, order[cl_id]]
            real_values = background_np[:, order[cl_id]]
            add_scatter_trace(fig, real_values, class_shap_values, cl)
        add_layout(fig, f"Methylation level", f"SHAP values", f"{feat}")
        fig.update_layout({'colorway': px.colors.qualitative.Set1})
        Path(f"features/scatter").mkdir(parents=True, exist_ok=True)
        save_figure(fig, f"features/scatter/{feat_id}_{feat}")

        fig = go.Figure()
        for cl_id, cl in enumerate(class_names):
            add_violin_trace(fig, common_df.loc[common_df['Status'] == cl_id, feat].values, cl)
        add_layout(fig, "", f"{feat}", f"")
        fig.update_layout({'colorway': px.colors.qualitative.Set1})
        Path(f"features/violin").mkdir(parents=True, exist_ok=True)
        save_figure(fig, f"features/violin/{feat_id}_{feat}")

    for cl_id, cl in enumerate(class_names):
        class_shap_values = shap_values[cl_id]
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
        df_features.to_excel(f"{cl}/shap.xlsx", index=False)


if __name__ == "__main__":
    main()
