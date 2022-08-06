import torch
from src.models.tabular.base import BaseModel
from src.models.tabular.nbm_spam.archs.concept_nam import ConceptNAMNary


class NAMModel(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.build_network()

    def build_network(self):
        self.model = ConceptNAMNary(
            num_concepts=self.hparams.input_dim,
            num_classes=self.hparams.output_dim,
            nary=self.hparams.nary,
            first_layer=self.hparams.first_layer,
            first_hidden_dim=self.hparams.first_hidden_dim,
            hidden_dims=self.hparams.hidden_dims,
            num_subnets=self.hparams.num_subnets,
            dropout=self.hparams.dropout,
            concept_dropout=self.hparams.concept_dropout,
            batchnorm=self.hparams.batchnorm,
            output_penalty=self.hparams.output_penalty,
            polynomial=self.hparams.polynomial
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
