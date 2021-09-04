import torchmetrics
from pytorch_tabnet.metrics import Metric


class custom_metric(Metric):
    def __init__(self, metric_type, num_classes, average):
        if average != '':
            self._name = f"{metric_type}_{average}"
        else:
            self._name = f"{metric_type}"
        self._maximize = True
        self.metric_type = metric_type
        self.num_classes = num_classes
        self.average = average

    def __call__(self, y_true, y_score):
        if self.metric_type == "accuracy":
            value = torchmetrics.functional.accuracy(y_score, y_true, average=self.average, num_classes=self.num_classes)
        elif self.metric_type == "f1":
            value = torchmetrics.functional.accuracy(y_score, y_true, average=self.average, num_classes=self.num_classes)
        elif self.metric_type == "precision":
            value = torchmetrics.functional.precision(y_score, y_true, average=self.average, num_classes=self.num_classes)
        elif self.metric_type == "recall":
            value = torchmetrics.functional.recall(y_score, y_true, average=self.average, num_classes=self.num_classes)
        elif self.metric_type == "cohen_kappa":
            value = torchmetrics.functional.cohen_kappa(y_score, y_true, num_classes=self.num_classes)
        elif self.metric_type == "matthews_corrcoef":
            value = torchmetrics.functional.matthews_corrcoef(y_score, y_true, num_classes=self.num_classes)
        else:
            raise ValueError("Unsupported metrics")
        return value