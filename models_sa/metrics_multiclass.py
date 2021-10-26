import torchmetrics
import torch


def init_metric(self):
    if self.average != '':
        self._name = f"{self.metric_type}_{self.average}"
    else:
        self._name = f"{self.metric_type}"
    self._maximize = True


def calc_metric(self, y_true, y_score):
    y_true_torch = torch.from_numpy(y_true)
    y_score_torch = torch.from_numpy(y_score)
    if self.metric_type == "accuracy":
        value = torchmetrics.functional.accuracy(y_score_torch, y_true_torch, average=self.average, num_classes=self.num_classes)
    elif self.metric_type == "f1":
        value = torchmetrics.functional.accuracy(y_score_torch, y_true_torch, average=self.average, num_classes=self.num_classes)
    elif self.metric_type == "precision":
        value = torchmetrics.functional.precision(y_score_torch, y_true_torch, average=self.average, num_classes=self.num_classes)
    elif self.metric_type == "recall":
        value = torchmetrics.functional.recall(y_score_torch, y_true_torch, average=self.average, num_classes=self.num_classes)
    elif self.metric_type == "cohen_kappa":
        value = torchmetrics.functional.cohen_kappa(y_score_torch, y_true_torch, num_classes=self.num_classes)
    elif self.metric_type == "matthews_corrcoef":
        value = torchmetrics.functional.matthews_corrcoef(y_score_torch, y_true_torch, num_classes=self.num_classes)
    else:
        raise ValueError("Unsupported metrics")
    value = float(value.numpy())
    return value


def get_metrics_dict(num_classes, base_class):
    d = {
        "accuracy_macro": type(
            "accuracy_macro",
            (base_class,),
            {
                "metric_type": "accuracy",
                "average": "macro",
                "num_classes": num_classes,
                "__init__": init_metric,
                "__call__": calc_metric
            }
        ),
        "accuracy_weighted": type(
            "accuracy_weighted",
            (base_class,),
            {
                "metric_type": "accuracy",
                "average": "weighted",
                "num_classes": num_classes,
                "__init__": init_metric,
                "__call__": calc_metric
            }
        ),
        "f1_macro": type(
            "f1_macro",
            (base_class,),
            {
                "metric_type": "f1",
                "average": "macro",
                "num_classes": num_classes,
                "__init__": init_metric,
                "__call__": calc_metric
            }
        ),
        "f1_weighted": type(
            "f1_weighted",
            (base_class,),
            {
                "metric_type": "f1",
                "average": "weighted",
                "num_classes": num_classes,
                "__init__": init_metric,
                "__call__": calc_metric
            }
        ),
        "cohen_kappa": type(
            "cohen_kappa",
            (base_class,),
            {
                "metric_type": "cohen_kappa",
                "average": "",
                "num_classes": num_classes,
                "__init__": init_metric,
                "__call__": calc_metric
            }
        ),
        "matthews_corrcoef": type(
            "matthews_corrcoef",
            (base_class,),
            {
                "metric_type": "matthews_corrcoef",
                "average": "",
                "num_classes": num_classes,
                "__init__": init_metric,
                "__call__": calc_metric
            }
        )
    }

    return d
