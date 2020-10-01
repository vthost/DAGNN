from sklearn.metrics import *
import numpy as np
import warnings
import torch
import statistics


def report(labels, predictions, classification=True):

    if not classification:
        y_true = labels.numpy()
        y_pred = predictions.numpy()

        return {"mae": mean_absolute_error(y_true, y_pred),
                "mse": mean_squared_error(y_true, y_pred, squared=True),
                "rmse": mean_squared_error(y_true, y_pred, squared=False)}

    # if standard binary labels (= all but IPC)
    if len(predictions.shape) <= 1 or predictions.shape[1] == 2:

        # print(labels, predictions)
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        if isinstance(predictions, torch.Tensor):
            if len(predictions.shape) > 1:
                predictions_orig = predictions[:, 0]
            predictions = predictions.numpy()
        # print(predictions)
        if len(predictions.shape) > 1:  # one-hot
            predictions = np.argmax(predictions, axis=1)
            predictions_orig = predictions_orig.numpy()
            # hack
            # labels = np.argmax(labels, axis=1)
        else:
            predictions_orig = predictions
            predictions = predictions.round()
        # print(predictions)
        # print(labels)

        # print(sum(labels), sum(predictions))
        pres, recs, thresholds = precision_recall_curve(labels, predictions_orig)
        f1, pre, rec = (0, 0, 0) if sum(predictions) == 0 else \
            (f1_score(labels, predictions),
             precision_score(labels, predictions),
             recall_score(labels, predictions))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mcc = matthews_corrcoef(labels, predictions)

        return {"acc": accuracy_score(labels, predictions),
                "f1": f1, "pre": pre, "rec": rec,
                "mcc": mcc,
                "roc_auc": roc_auc_score(labels, predictions_orig),
                # Area Under the Receiver Operating Characteristic Curve
                "pr_auc": auc(recs, pres),  # area under precision-recall curve
                "avg_pre": average_precision_score(labels, predictions_orig),
                }

    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.numpy()

    predictions1 = np.argmax(predictions, axis=1)
    labels1 = labels[np.arange(labels.shape[0]), predictions1]


    predictions = predictions.round()
    # else ipc... "reuse acc label.."
    return {"acc": accuracy_score(labels1, predictions1),
            "f1-mac": f1_score(labels, predictions, average="macro"),
            "f1-mic": f1_score(labels, predictions, average="micro")
            # "roc_auc": 0, "pre": 0, "rec": 0,
            # "pr_auc": 0,
            # "avg_pre": 0,
            }


def report_to_str(metrics, keys=True):
    if keys:
        return "\t".join([k.upper() + (': {:.2f}'.format(v*100) if k != "loss" else ': {:.4f}'.format(v)) for k, v in metrics.items()])

    return ",".join(['{:.2f}'.format(v*100) if k != "loss" else '{:.4f}'.format(v) for k, v in metrics.items()])


def report_to_str_report(metrics):
    return ['{:.2f}'.format(v*100) if k != "loss" else '{:.4f}'.format(v) for k, v in metrics.items()]


def summary_report(metrics):
    result = {}
    for k, v in metrics.items():
        result[k] = sum(v)/len(v)
        if "acc" in k or "f1" in k or "mae" in k or "mse" in k or "rmse" in k:
            result[k+"-stdev"] = statistics.stdev(v)
    return result
