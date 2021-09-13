import numpy as np
from src.models.dnam.tabnet import TabNetModel
import torch
import os
import matplotlib.pyplot as plt
import pickle
from src.datamodules.datasets.dnam_dataset import DNAmDataset
from torch.utils.data import DataLoader
from scipy.sparse import csc_matrix
from scipy.special import log_softmax
from scipy.special import softmax
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


log = utils.get_logger(__name__)

dotenv.load_dotenv(override=True)

@hydra.main(config_path="../configs/", config_name="main_xai.yaml")
def main(config: DictConfig):

    class_names = [
        "Control",
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

    model = TabNetModel.load_from_checkpoint(checkpoint_path=f"{checkpoint_path}/{checkpoint_name}")
    model.produce_probabilities = True
    model.eval()
    model.freeze()

    # Init Lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule.setup()

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

    background_dataloader = test_dataloader

    feature_importances = np.zeros((model.hparams.input_dim))
    for data, targets, indexes in tqdm(background_dataloader):
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

    background, outs_real, indexes = next(iter(background_dataloader))
    background_np = background.cpu().detach().numpy()
    outs_real_np = outs_real.cpu().detach().numpy()

    deep_explainer = shap.DeepExplainer(model, background)
    shap_values = deep_explainer.shap_values(background)

    shap.summary_plot(
        shap_values=shap_values,
        features=background_np,
        feature_names=datamodule.betas.columns.values,
        max_display=num_top_features,
        class_names=class_names,
        class_inds=[0, 1, 2, 3],
        plot_size=(12, 8),
        show=False
    )
    plt.savefig('bar.png')
    plt.savefig('bar.pdf')
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
            subj_global_id = indexes[subj_id]
            subj_name = datamodule.betas.index.values[subj_global_id]
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[subj_cl][subj_id],
                    base_values=deep_explainer.expected_value[subj_cl],
                    data=background_np[subj_id],
                    feature_names=datamodule.betas.columns.values
                ),
                max_display=num_top_features,
                show=False
            )
            fig = plt.gcf()
            fig.set_size_inches(18, 8, forward=True)
            fig.savefig(f"{class_names[subj_cl]}/waterfall_{passed_examples[subj_cl]}_{subj_name}.pdf")
            fig.savefig(f"{class_names[subj_cl]}/waterfall_{passed_examples[subj_cl]}_{subj_name}.png")
            plt.close()
            passed_examples[subj_cl] += 1

    for cl_id, cl in enumerate(class_names):
        class_shap_values = shap_values[cl_id]
        shap_abs = np.absolute(class_shap_values)
        shap_mean_abs = np.mean(shap_abs, axis=0)
        order = np.argsort(shap_mean_abs)[::-1]
        features = datamodule.betas.columns.values
        features_best = features[order[0:num_top_features]]
        for feat in features_best:
            shap.plots.scatter(
                shap.Explanation(
                    values=shap_values[cl_id],
                    base_values=deep_explainer.expected_value,
                    data=background_np,
                    feature_names=datamodule.betas.columns.values
                )[:, feat],
                alpha=0.6,
                hist=True,
                color=outs_real_np,
                show=False,
                cmap=plt.get_cmap("Set1"),
            )
            fig = plt.gcf()
            fig.set_size_inches(18, 8, forward=True)
            plt.savefig(f"{cl}/scatter_{feat}.png")
            plt.savefig(f"{cl}/scatter_{feat}.pdf")
            plt.close()
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
