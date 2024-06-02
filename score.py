import numpy as np
import scipy
from sklearn import metrics


def get_logits(logits: np.ndarray, labels: np.ndarray, stable=True):
    """Extract numerically (or not) stable logits.
    
    Inputs:
        logits: [*, n_classes] - initial logits.
        labels: [*, 1] - correct predictions.
    Returns:
        logits: [*] - logits for ground-truth classes.
    """
    sz = logits.shape[:-1]

    probabilities = logits - np.max(logits, axis=-1, keepdims=True)
    probabilities = np.array(np.exp(probabilities), np.float64)
    probabilities = probabilities / np.sum(probabilities, axis=-1, keepdims=True)

    probabilities = probabilities.reshape(-1, logits.shape[-1])
    y_true = probabilities[np.arange(probabilities.shape[0]), labels.reshape(-1)]
    
    if stable:
        probabilities[np.arange(probabilities.shape[0]), labels.reshape(-1)] = 0
        y_wrong = np.sum(probabilities, axis=-1)

        logit = np.log(y_true + 1e-45) - np.log(y_wrong + 1e-45)
    else:
        logit = np.log(y_true + 1e-45) - np.log(1 - y_true + 1e-45)
    logit = logit.reshape(*sz)

    return logit


def get_hinge(logits: np.ndarray, labels: np.ndarray):
    """Extract hinge.
    
    Inputs:
        logits: [*, n_classes] - initial logits.
        labels: [*, 1] - correct predictions.
    Returns:
        logits: [*] - hinge logits for ground-truth classes.
    """
    sz = logits.shape[:-1]

    logits = logits.reshape(-1, logits.shape[-1])
    logits_true = logits[np.arange(logits.shape[0]), labels.reshape(-1)]
    logits[np.arange(logits.shape[0]), labels.reshape(-1)] = np.min(logits)

    logit = logits_true - np.max(logits)
    logit = logit.reshape(*sz)

    return logit


def sweep(score, x):
    fpr, tpr, _ = metrics.roc_curve(x, -score, pos_label=1)
    acc = np.max(1-(fpr+(1-tpr))/2)
    return fpr, tpr, metrics.auc(fpr, tpr), acc


def lira_offline(target_scores: np.ndarray, shadow_scores: np.ndarray, labels: np.ndarray,
                 fix_variance: bool = False):
    """Score offline using LiRA approach: https://arxiv.org/pdf/2112.03570.pdf.

    target_scores: [n_examples, n_aug]
    shadow_scores: [n_examples, n_shadow, n_aug]
    labels: [n_examples]    
    """
    mean_out = np.median(shadow_scores, 1)

    if fix_variance:
        std_out = np.std(shadow_scores)
    else:
        std_out = np.std(shadow_scores, 1)
    
    # [n_examples, n_aug], [n_examples, n_aug]
    score = scipy.stats.norm.logpdf(target_scores, mean_out, std_out+1e-30)
    predictions = np.array(score.mean(1))
    fpr, tpr, auc, acc = sweep(np.array(predictions), labels.astype(bool))

    fpr_rates = [.0001, .001, .01, .1]
    lows = np.array([tpr[np.where(fpr<rate)[0][-1]] for rate in fpr_rates])

    return fpr, tpr, auc, acc, lows


def calibration(target_scores: np.ndarray, shadow_scores: np.ndarray, labels: np.ndarray):
    """Score using difficulty calibration approach: https://arxiv.org/pdf/2111.08440.pdf
    
    target_scores: [n_examples, n_aug]
    shadow_scores: [n_examples, n_shadow, n_aug]
    labels: [n_examples]    
    """
    score = target_scores - np.median(shadow_scores, 1)
    predictions = np.array(score.mean(1))
    fpr, tpr, auc, acc = sweep(np.array(predictions), labels.astype(bool))
    
    fpr_rates = [.0001, .001, .01, .1]
    lows = np.array([tpr[np.where(fpr<rate)[0][-1]] for rate in fpr_rates])

    return fpr, tpr, auc, acc, lows
