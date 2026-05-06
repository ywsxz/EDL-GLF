import random
import pandas as pd
import torch
from medpy.metric import sensitivity
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, cohen_kappa_score, balanced_accuracy_score
from torchmetrics.functional.classification import multiclass_specificity
import numpy as np
from medpy import metric


def save_result_to_csv(info_dict, save_path):
    result_dict = {
        'image_name': info_dict['images_name'],
        'label': info_dict['ground_truth'],
        'label_pred': info_dict['predicted_class'],
    }
    class_num = info_dict['predicted_softmax'].shape[-1]
    for i in range(class_num):
        key_ = f'label_{i}_proba'
        result_dict[key_] = info_dict['predicted_softmax'][:, i].tolist()

    result_df = pd.DataFrame(result_dict)
    result_df.to_csv(save_path, index=False)


def calculate_binary_classification_metric(preds, gts):
    preds = np.array(preds).reshape(-1)
    gts = np.array(gts).reshape(-1)
    confusion = confusion_matrix(gts, preds)
    TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]
    accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
    sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
    specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
    precision = float(TP) / float(TP + FP) if float(TP + FP) != 0 else 0
    f1 = float(2 * precision * sensitivity) / float(precision + sensitivity) if float(
        precision + sensitivity) != 0 else 0

    metric_dict = {'accuracy': accuracy,
                   'sensitivity': sensitivity,
                   'specificity': specificity,
                   'precision': precision,
                   'f1_score': f1,
                   'confusion_matrix': confusion}
    return metric_dict


def calculate_multi_classification_metric(preds, gts, num_classes, average='macro'):
    preds = np.array(preds).reshape(-1)
    gts = np.array(gts).reshape(-1)
    confusion = confusion_matrix(gts, preds)
    accuracy = accuracy_score(gts, preds)
    sensitivity_recall = recall_score(gts, preds, average=average)
    specificity = multiclass_specificity(torch.tensor(preds), torch.tensor(gts), num_classes=num_classes, average=average)
    precision = precision_score(gts, preds, average=average)
    f1 = f1_score(gts, preds, average=average)
    kappa_score = cohen_kappa_score(gts, preds, weights='quadratic')  # linear, quadratic
    bmca = balanced_accuracy_score(gts, preds) 

    metric_dict = {'accuracy': accuracy,
                   'sensitivity': sensitivity_recall,
                   'specificity': specificity.item(),
                   'precision': precision,
                   'f1_score': f1,
                   'cohen_kappa_score': kappa_score,
                   'confusion_matrix': confusion,
                   'balanced_accuracy': bmca}
    return metric_dict

def calculate_ece_from_probs(probs, labels, n_bins=10):
    probs = np.asarray(probs)
    labels = np.asarray(labels).reshape(-1).astype(np.int64)

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels

    ece = 0.0
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)

    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        if i == 0:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences > lower) & (confidences <= upper)

        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin

    return float(ece)

def calculate_brier_score_from_probs(probs, labels, num_classes):
    probs = np.asarray(probs)
    labels = np.asarray(labels).reshape(-1).astype(np.int64)

    one_hot_labels = np.eye(num_classes)[labels]
    brier_score = np.mean(np.sum((probs - one_hot_labels) ** 2, axis=1))

    return float(brier_score)
def split_list(full_list, ratio=None, shuffle=True):

    if ratio is None:
        ratio = [0.8, 0.2]

    if len(ratio) == 1:
        ratio.append(0.0)

    assert sum(ratio) <= 1, 'The ratio sum must be less than 1'

    sub_list_num = len(ratio)
    list_num = len(full_list)

    if shuffle:
        random.shuffle(full_list)

    sub_lists = []

    count = 0
    for i in range(sub_list_num):
        elem_num = round(list_num * ratio[i])
        if (i + 1) == sub_list_num:
            sub_lists.append(full_list[count:])
        else:
            sub_lists.append(full_list[count:(count + elem_num)])
        count = count + elem_num

    return sub_lists
