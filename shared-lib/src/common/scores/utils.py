import numpy as np
import segmentation_models_pytorch as smp
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def scalar_metrics(all_labels: np.array, all_preds: np.array):
    """
    Computes scalar scores between predictions and ground truth
    """
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    return accuracy, precision, recall, f1


def mask_metrics(pred_mask: torch.Tensor, label_mask: torch.Tensor, threshold: float = 0.5):
    """
    Computes mask scores between predictions and ground truth
    """
    pred_flattened = pred_mask.flatten()
    pred_flattened = pred_flattened[:, None]
    label_flattened = label_mask.flatten().long()
    label_flattened = label_flattened[:, None]
    tp, fp, fn, tn = smp.metrics.get_stats(
        pred_flattened, label_flattened, mode="binary", threshold=threshold
    )
    accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
    precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
    recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
    f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
    return accuracy, precision, recall, f1
