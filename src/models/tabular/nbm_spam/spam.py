import torch
from src.models.tabular.base import BaseModel
from src.models.tabular.nbm_spam.archs.concept_spam import ConceptSPAM


class SPAMModel(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.build_network()

    def build_network(self):
        self.model = ConceptSPAM(
            num_concepts=self.hparams.input_dim,
            num_classes=self.hparams.output_dim,
            ranks=self.hparams.ranks,
            dropout=self.hparams.dropout,
            ignore_unary=self.hparams.ignore_unary,
            reg_order=self.hparams.reg_order,
            lower_order_correction=self.hparams.lower_order_correction,
            use_geometric_mean=self.hparams.use_geometric_mean,
            orthogonal=self.hparams.orthogonal,
            proximal=self.hparams.proximal,
        )

    def forward(self, batch):
        if isinstance(batch, dict):
            x = batch["all"]
        else:
            x = batch[:, self.feats_all_ids]
        x = self.model(x)
        if isinstance(x, tuple):
            x = x[0]
        if self.produce_probabilities:
            return torch.softmax(x, dim=1)
        else:
            return x
