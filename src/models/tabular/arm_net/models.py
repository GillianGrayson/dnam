import torch
from src.models.tabular.base import BaseModel
from .repository.models.lr import LRModel
from .repository.models.fm import FMModel
from .repository.models.hofm import HOFMModel
from .repository.models.afm import AFMModel
from .repository.models.dcn import CrossNetModel
from .repository.models.xdfm import CINModel
from .repository.models.dnn import DNNModel
from .repository.models.gcn import GCNModel
from .repository.models.gat import GATModel
from .repository.models.wd import WDModel
from .repository.models.pnn import IPNNModel
from .repository.models.pnn import KPNNModel
from .repository.models.nfm import NFMModel
from .repository.models.dfm import DeepFMModel
from .repository.models.dcn import DCNModel
from .repository.models.xdfm import xDeepFMModel
from .repository.models.afn import AFNModel
from .repository.models.armnet import ARMNetModel
from .repository.models.armnet_1h import ARMNetModel as ARMNet1H
from .repository.models.gc_arm import GC_ARMModel
from .repository.models.sa_glu import SA_GLUModel
import numpy as np


class ARMNetModels(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.build_network()

    def build_network(self):
        if self.hparams.model == 'lr':
            self.model = LRModel(self.hparams.nfeat)
        elif self.hparams.model == 'fm':
            self.model = FMModel(self.hparams.nfeat, self.hparams.nemb)
        elif self.hparams.model == 'hofm':
            self.model = HOFMModel(self.hparams.nfeat, self.hparams.nemb, self.hparams.k)
        elif self.hparams.model == 'afm':
            self.model = AFMModel(self.hparams.nfeat, self.hparams.nemb, self.hparams.h, self.hparams.dropout)
        elif self.hparams.model == 'dcn':
            self.model = CrossNetModel(self.hparams.nfield, self.hparams.nfeat, self.hparams.nemb, self.hparams.k)
        elif self.hparams.model == 'cin':
            self.model = CINModel(self.hparams.nfield, self.hparams.nfeat, self.hparams.nemb, self.hparams.k, self.hparams.h)
        elif self.hparams.model == 'afn':
            self.model = AFNModel(self.hparams.nfield, self.hparams.nfeat, self.hparams.nemb, self.hparams.h, self.hparams.mlp_nlayer, self.hparams.mlp_nhid, self.hparams.dropout, self.hparams.ensemble, self.hparams.dnn_nlayer, self.hparams.dnn_nhid)
        elif self.hparams.model == 'armnet':
            self.model = ARMNetModel(
                nfield=self.hparams.nfield,
                nfeat=self.hparams.nfeat,
                nemb=self.hparams.nemb,
                nhead=self.hparams.nattn_head,
                alpha=self.hparams.alpha,
                nhid=self.hparams.h,
                mlp_nlayer=self.hparams.mlp_nlayer,
                mlp_nhid=self.hparams.mlp_nhid,
                dropout=self.hparams.dropout,
                ensemble=self.hparams.ensemble,
                deep_nlayer=self.hparams.dnn_nlayer,
                deep_nhid=self.hparams.dnn_nhid,
                noutput=self.hparams.output_dim
            )
        elif self.hparams.model == 'armnet_1h':
            self.model = ARMNet1H(self.hparams.nfield, self.hparams.nfeat, self.hparams.nemb, self.hparams.alpha, self.hparams.h, self.hparams.nemb, self.hparams.mlp_nlayer, self.hparams.mlp_nhid, self.hparams.dropout, self.hparams.ensemble, self.hparams.dnn_nlayer, self.hparams.dnn_nhid)
        elif self.hparams.model == 'dnn':
            self.model = DNNModel(self.hparams.nfield, self.hparams.nfeat, self.hparams.nemb, self.hparams.mlp_nlayer, self.hparams.mlp_nhid, self.hparams.dropout)
        elif self.hparams.model == 'gcn':
            self.model = GCNModel(self.hparams.nfield, self.hparams.nfeat, self.hparams.nemb, self.hparams.k, self.hparams.h, self.hparams.mlp_nlayer, self.hparams.mlp_nhid, self.hparams.dropout)
        elif self.hparams.model == 'gat':
            self.model = GATModel(self.hparams.nfield, self.hparams.nfeat, self.hparams.nemb, self.hparams.k, self.hparams.h, self.hparams.mlp_nlayer, self.hparams.mlp_nhid, self.hparams.dropout, 0.2, self.hparams.nattn_head)
        elif self.hparams.model == 'wd':
            self.model = WDModel(self.hparams.nfield, self.hparams.nfeat, self.hparams.nemb, self.hparams.mlp_nlayer, self.hparams.mlp_nhid, self.hparams.dropout)
        elif self.hparams.model == 'ipnn':
            self.model = IPNNModel(self.hparams.nfield, self.hparams.nfeat, self.hparams.nemb, self.hparams.mlp_nlayer, self.hparams.mlp_nhid, self.hparams.dropout)
        elif self.hparams.model == 'kpnn':
            self.model = KPNNModel(self.hparams.nfield, self.hparams.nfeat, self.hparams.nemb, self.hparams.mlp_nlayer, self.hparams.mlp_nhid, self.hparams.dropout)
        elif self.hparams.model == 'nfm':
            self.model = NFMModel(self.hparams.nfeat, self.hparams.nemb, self.hparams.mlp_nlayer, self.hparams.mlp_nhid, self.hparams.dropout)
        elif self.hparams.model == 'dfm':
            self.model = DeepFMModel(self.hparams.nfield, self.hparams.nfeat, self.hparams.nemb, self.hparams.mlp_nlayer, self.hparams.mlp_nhid, self.hparams.dropout)
        elif self.hparams.model == 'dcn+':
            self.model = DCNModel(self.hparams.nfield, self.hparams.nfeat, self.hparams.nemb, self.hparams.k, self.hparams.mlp_nlayer, self.hparams.mlp_nhid, self.hparams.dropout)
        elif self.hparams.model == 'xdfm':
            self.model = xDeepFMModel(self.hparams.nfield, self.hparams.nfeat, self.hparams.nemb, self.hparams.k, self.hparams.h, self.hparams.mlp_nlayer, self.hparams.mlp_nhid, self.hparams.dropout)
        elif self.hparams.model == 'gc_arm':
            self.model = GC_ARMModel(self.hparams.nfield, self.hparams.nfeat, self.hparams.nemb, self.hparams.nattn_head, self.hparams.alpha, self.hparams.h, self.hparams.mlp_nlayer, self.hparams.mlp_nhid, self.hparams.dropout, self.hparams.ensemble, self.hparams.dnn_nlayer, self.hparams.dnn_nhid)
        elif self.hparams.model == 'sa_glu':
            self.model = SA_GLUModel(self.hparams.nfield, self.hparams.nfeat, self.hparams.nemb, self.hparams.mlp_nlayer, self.hparams.mlp_nhid, self.hparams.dropout, self.hparams.ensemble, self.hparams.dnn_nlayer, self.hparams.dnn_nhid)
        else:
            raise ValueError(f'Unsupported model: {self.hparams.model}')

    def forward(self, batch):
        if isinstance(batch, dict):
            batch_size = batch["all"].shape[0]
            feats_ids = torch.LongTensor(np.arange(self.hparams.nfeat)).to(batch['all'].device)
            feats_ids = feats_ids.repeat(batch["all"].shape[0], 1)
            x = {
                'id': feats_ids,
                'value': batch["all"],
            }
        else:
            batch_size = batch.shape[0]
            feats_ids = torch.LongTensor(np.arange(self.hparams.nfeat)).to(batch.device)
            feats_ids = feats_ids.repeat(batch.shape[0], 1)
            x = {
                'id': feats_ids,
                'value': batch,
            }
        x = self.model(x)
        x = x.view(batch_size, -1)
        if self.produce_probabilities:
            return torch.softmax(x, dim=1)
        else:
            return x
