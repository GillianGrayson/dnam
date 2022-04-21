from typing import Any, List
from torch import nn
from torchmetrics import MetricCollection, Accuracy, F1, Precision, Recall, CohenKappa, MatthewsCorrcoef, AUROC
from torchmetrics import CosineSimilarity, MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError, PearsonCorrcoef, R2Score, SpearmanCorrcoef
import wandb
from typing import Dict
import pytorch_lightning as pl
import torch
from .blocks import *
from pytorch_tabnet.utils import create_explain_matrix


class SaintModel(pl.LightningModule):

    def __init__(
            self,
            task="regression",
            input_dim=100,
            output_dim=4,

            categories=None,
            num_continuous=100,
            dim=32,
            depth=6,
            heads=8,
            dim_head=16,
            dim_out=1,
            mlp_hidden_mults=(4, 2),
            mlp_act=None,
            num_special_tokens=0,
            attn_dropout=0.1,
            ff_dropout=0.1,
            cont_embeddings='MLP', # ['MLP','Noemb','pos_singleMLP']
            scalingfactor=10,
            attentiontype='col', # ['col','colrow','row','justmlp','attn','attnmlp']
            final_mlp_style='common',

            loss_type="MSE",
            optimizer_lr=0.0001,
            optimizer_weight_decay=0.0005,
            scheduler_step_size=50,
            scheduler_gamma=0.9,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self._build_network()

        self.ids_cat = None
        self.ids_con = None

        self.task = task
        self.produce_probabilities = False
        self.produce_importance = False

        if task == "classification":
            self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
            if output_dim < 2:
                raise ValueError(f"Classification with {output_dim} classes")
            self.metrics_dict = {
                'accuracy_macro': Accuracy(num_classes=self.hparams.output_dim, average='macro'),
                'accuracy_micro': Accuracy(num_classes=self.hparams.output_dim, average='micro'),
                'accuracy_weighted': Accuracy(num_classes=self.hparams.output_dim, average='weighted'),
                'f1_macro': F1(num_classes=self.hparams.output_dim, average='macro'),
                'f1_micro': F1(num_classes=self.hparams.output_dim, average='micro'),
                'f1_weighted': F1(num_classes=self.hparams.output_dim, average='weighted'),
                'precision_macro': Precision(num_classes=self.hparams.output_dim, average='macro'),
                'precision_micro': Precision(num_classes=self.hparams.output_dim, average='micro'),
                'precision_weighted': Precision(num_classes=self.hparams.output_dim, average='weighted'),
                'recall_macro': Recall(num_classes=self.hparams.output_dim, average='macro'),
                'recall_micro': Recall(num_classes=self.hparams.output_dim, average='micro'),
                'recall_weighted': Recall(num_classes=self.hparams.output_dim, average='weighted'),
                'cohens_kappa': CohenKappa(num_classes=self.hparams.output_dim),
                'matthews_corr': MatthewsCorrcoef(num_classes=self.hparams.output_dim),
            }
            self.metrics_summary = {
                'accuracy_macro': 'max',
                'accuracy_micro': 'max',
                'accuracy_weighted': 'max',
                'f1_macro': 'max',
                'f1_micro': 'max',
                'f1_weighted': 'max',
                'precision_macro': 'max',
                'precision_micro': 'max',
                'precision_weighted': 'max',
                'recall_macro': 'max',
                'recall_micro': 'max',
                'recall_weighted': 'max',
                'cohens_kappa': 'max',
                'matthews_corr': 'max',
            }
            self.metrics_prob_dict = {
                'auroc_macro': AUROC(num_classes=self.hparams.output_dim, average='macro'),
                'auroc_micro': AUROC(num_classes=self.hparams.output_dim, average='micro'),
                'auroc_weighted': AUROC(num_classes=self.hparams.output_dim, average='weighted'),
            }
            self.metrics_prob_summary = {
                'auroc_macro': 'max',
                'auroc_micro': 'max',
                'auroc_weighted': 'max',
            }
        elif task == "regression":
            if self.hparams.loss_type == "MSE":
                self.loss_fn = torch.nn.MSELoss(reduction='mean')
            elif self.hparams.loss_type == "L1Loss":
                self.loss_fn = torch.nn.L1Loss(reduction='mean')
            else:
                raise ValueError("Unsupported loss_type")
            self.metrics_dict = {
                'CosineSimilarity': CosineSimilarity(),
                'MeanAbsoluteError': MeanAbsoluteError(),
                'MeanAbsolutePercentageError': MeanAbsolutePercentageError(),
                'MeanSquaredError': MeanSquaredError(),
                'PearsonCorrcoef': PearsonCorrcoef(),
                'R2Score': R2Score(),
                'SpearmanCorrcoef': SpearmanCorrcoef(),
            }
            self.metrics_summary = {
                'CosineSimilarity': 'min',
                'MeanAbsoluteError': 'min',
                'MeanAbsolutePercentageError': 'min',
                'MeanSquaredError': 'min',
                'PearsonCorrcoef': 'max',
                'R2Score': 'max',
                'SpearmanCorrcoef': 'max'
            }
            self.metrics_prob_dict = {}
            self.metrics_prob_summary = {}

        self.metrics_train = MetricCollection(self.metrics_dict)
        self.metrics_train_prob = MetricCollection(self.metrics_prob_dict)
        self.metrics_val = self.metrics_train.clone()
        self.metrics_val_prob = self.metrics_train_prob.clone()
        self.metrics_test = self.metrics_train.clone()
        self.metrics_test_prob = self.metrics_train_prob.clone()

    def _build_network(self):
        assert all(map(lambda n: n > 0, self.hparams.categories)), 'number of each category must be positive'

        # categories related calculations

        self.num_categories = len(self.hparams.categories)
        self.num_unique_categories = sum(self.hparams.categories)

        # create category embeddings table

        self.num_special_tokens = self.hparams.num_special_tokens
        self.total_tokens = self.num_unique_categories + self.hparams.num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        categories_offset = F.pad(torch.tensor(list(self.hparams.categories)), (1, 0), value=self.hparams.num_special_tokens)
        categories_offset = categories_offset.cumsum(dim=-1)[:-1]

        self.register_buffer('categories_offset', categories_offset)

        self.norm = nn.LayerNorm(self.hparams.num_continuous)
        self.num_continuous = self.hparams.num_continuous
        self.dim = self.hparams.dim
        self.cont_embeddings = self.hparams.cont_embeddings
        self.attentiontype = self.hparams.attentiontype
        self.final_mlp_style = self.hparams.final_mlp_style

        if self.cont_embeddings == 'MLP':
            self.simple_MLP = nn.ModuleList([simple_MLP([1, 100, self.dim]) for _ in range(self.num_continuous)])
            input_size = (self.hparams.dim * self.num_categories) + (self.hparams.dim * self.hparams.num_continuous)
            nfeats = self.num_categories + self.hparams.num_continuous
        elif self.cont_embeddings == 'pos_singleMLP':
            self.simple_MLP = nn.ModuleList([simple_MLP([1, 100, self.dim]) for _ in range(1)])
            input_size = (self.hparams.dim * self.num_categories) + (self.hparams.dim * self.hparams.num_continuous)
            nfeats = self.num_categories + self.hparams.num_continuous
        else:
            print('Continous features are not passed through attention')
            input_size = (self.hparams.dim * self.num_categories) + self.hparams.num_continuous
            nfeats = self.num_categories

            # transformer
        if self.hparams.attentiontype == 'col':
            self.transformer = Transformer(
                num_tokens=self.total_tokens,
                dim=self.hparams.dim,
                depth=self.hparams.depth,
                heads=self.hparams.heads,
                dim_head=self.hparams.dim_head,
                attn_dropout=self.hparams.attn_dropout,
                ff_dropout=self.hparams.ff_dropout
            )
        elif self.hparams.attentiontype in ['row', 'colrow']:
            self.transformer = RowColTransformer(
                num_tokens=self.total_tokens,
                dim=self.hparams.dim,
                nfeats=nfeats,
                depth=self.hparams.depth,
                heads=self.hparams.heads,
                dim_head=self.hparams.dim_head,
                attn_dropout=self.hparams.attn_dropout,
                ff_dropout=self.hparams.ff_dropout,
                style=self.hparams.attentiontype
            )

        l = input_size // 8
        hidden_dimensions = list(map(lambda t: l * t, self.hparams.mlp_hidden_mults))
        all_dimensions = [input_size, *hidden_dimensions, self.hparams.dim_out]

        self.mlp = MLP(all_dimensions, act=self.hparams.mlp_act)
        self.embeds = nn.Embedding(self.total_tokens, self.dim)  # .to(device)

        cat_mask_offset = F.pad(torch.Tensor(self.num_categories).fill_(2).type(torch.int8), (1, 0), value=0)
        cat_mask_offset = cat_mask_offset.cumsum(dim=-1)[:-1]

        con_mask_offset = F.pad(torch.Tensor(self.num_continuous).fill_(2).type(torch.int8), (1, 0), value=0)
        con_mask_offset = con_mask_offset.cumsum(dim=-1)[:-1]

        self.register_buffer('cat_mask_offset', cat_mask_offset)
        self.register_buffer('con_mask_offset', con_mask_offset)

        self.mask_embeds_cat = nn.Embedding(self.num_categories * 2, self.dim)
        self.mask_embeds_cont = nn.Embedding(self.num_continuous * 2, self.dim)
        self.single_mask = nn.Embedding(2, self.dim)
        self.pos_encodings = nn.Embedding(self.num_categories + self.num_continuous, self.dim)

        if self.final_mlp_style == 'common':
            self.mlp1 = simple_MLP([self.hparams.dim, (self.total_tokens) * 2, self.total_tokens])
            self.mlp2 = simple_MLP([self.hparams.dim, (self.num_continuous), 1])

        else:
            self.mlp1 = sep_MLP(self.hparams.dim, self.num_categories, self.hparams.categories)
            self.mlp2 = sep_MLP(self.hparams.dim, self.num_continuous, np.ones(self.num_continuous).astype(int))

        self.mlpfory = simple_MLP([self.hparams.dim, 1000, self.hparams.output_dim])
        self.pt_mlp = simple_MLP([self.hparams.dim * (self.num_continuous + self.num_categories),
                                  6 * self.hparams.dim * (self.num_continuous + self.num_categories) // 5,
                                  self.hparams.dim * (self.num_continuous + self.num_categories) // 2])
        self.pt_mlp2 = simple_MLP([self.hparams.dim * (self.num_continuous + self.num_categories),
                                   6 * self.hparams.dim * (self.num_continuous + self.num_categories) // 5,
                                   self.hparams.dim * (self.num_continuous + self.num_categories) // 2])

    def forward(self, x):
        # Returns output and Masked Loss. We only need the output
        x_cat = x[:, self.hparams.ids_cat]
        x_con = x[:, self.hparams.ids_con]


    def on_fit_start(self) -> None:
        for stage_type in ['train', 'val', 'test']:
            for m, sum in self.metrics_summary.items():
                wandb.define_metric(f"{stage_type}/{m}", summary=sum)
            for m, sum in self.metrics_prob_summary.items():
                wandb.define_metric(f"{stage_type}/{m}", summary=sum)
            wandb.define_metric(f"{stage_type}/loss", summary='min')

    def step(self, batch: Any, stage:str):
        x, y, ind = batch
        out = self.forward(x)
        batch_size = x.size(0)
        if self.task == "regression":
            y = y.view(batch_size, -1)
        loss = self.loss_fn(out, y)

        logs = {"loss": loss}
        non_logs = {}
        if self.task == "classification":
            probs = torch.softmax(out, dim=1)
            preds = torch.argmax(out, dim=1)
            non_logs["preds"] = preds
            non_logs["targets"] = y
            if stage == "train":
                logs.update(self.metrics_train(preds, y))
                try:
                    logs.update(self.metrics_train_prob(probs, y))
                except ValueError:
                    pass
            elif stage == "val":
                logs.update(self.metrics_val(preds, y))
                try:
                    logs.update(self.metrics_val_prob(probs, y))
                except ValueError:
                    pass
            elif stage == "test":
                logs.update(self.metrics_test(preds, y))
                try:
                    logs.update(self.metrics_val_prob(probs, y))
                except ValueError:
                    pass
        elif self.task == "regression":
            if stage == "train":
                logs.update(self.metrics_train(out, y))
            elif stage == "val":
                logs.update(self.metrics_val(out, y))
            elif stage == "test":
                logs.update(self.metrics_test(out, y))

        return loss, logs, non_logs

    def training_step(self, batch: Any, batch_idx: int):
        loss, logs, non_logs = self.step(batch, "train")
        d = {f"train/{k}": v for k, v in logs.items()}
        self.log_dict(d, on_step=False, on_epoch=True, logger=True)
        logs.update(non_logs)
        return logs

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, logs, non_logs = self.step(batch, "val")
        d = {f"val/{k}": v for k, v in logs.items()}
        self.log_dict(d, on_step=False, on_epoch=True, logger=True)
        logs.update(non_logs)
        return logs

    def predict_step(self, batch, batch_idx):
        x, y, ind = batch
        out = self.forward(x)
        return out

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, logs, non_logs = self.step(batch, "test")
        d = {f"test/{k}": v for k, v in logs.items()}
        self.log_dict(d, on_step=False, on_epoch=True, logger=True)
        logs.update(non_logs)
        return logs

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.hparams.optimizer_lr,
            weight_decay=self.hparams.optimizer_weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=self.hparams.scheduler_step_size,
            gamma=self.hparams.scheduler_gamma
        )

        return (
            {
                "optimizer": optimizer,
                "lr_scheduler": scheduler
            }
        )
