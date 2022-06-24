from typing import Any, List
from .blocks import *
from src.models.base import BaseModel


class SaintModel(BaseModel):

    def __init__(
            self,
            task="regression",
            loss_type="MSE",
            input_dim=100,
            output_dim=1,
            optimizer_lr=0.0001,
            optimizer_weight_decay=0.0005,
            scheduler_step_size=50,
            scheduler_gamma=0.9,

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

            **kwargs
    ):
        super().__init__(
            task=task,
            loss_type=loss_type,
            input_dim=input_dim,
            output_dim=output_dim,
            optimizer_lr=optimizer_lr,
            optimizer_weight_decay=optimizer_weight_decay,
            scheduler_step_size=scheduler_step_size,
            scheduler_gamma=scheduler_gamma,
        )
        self.save_hyperparameters()
        self._build_network()

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

    def on_train_start(self) -> None:
        super().on_train_start()

    def on_fit_start(self) -> None:
        super().on_fit_start()

    def step(self, batch: Any, stage:str):
        return super().step(batch=batch, stage=stage)

    def training_step(self, batch: Any, batch_idx: int):
        return super().training_step(batch=batch, batch_idx=batch_idx)

    def training_epoch_end(self, outputs: List[Any]):
        return super().training_epoch_end(outputs=outputs)

    def validation_step(self, batch: Any, batch_idx: int):
        return super().validation_step(batch=batch, batch_idx=batch_idx)

    def validation_epoch_end(self, outputs: List[Any]):
        return super().validation_epoch_end(outputs=outputs)

    def test_step(self, batch: Any, batch_idx: int):
        return super().test_step(batch=batch, batch_idx=batch_idx)

    def test_epoch_end(self, outputs: List[Any]):
        return super().test_epoch_end(outputs=outputs)

    def predict_step(self, batch, batch_idx):
        return super().predict_step(batch=batch, batch_idx=batch_idx)

    def on_epoch_end(self):
        return super().on_epoch_end()

    def configure_optimizers(self):
        return super().configure_optimizers()
