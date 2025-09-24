# src/utils.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import os
import random
import numpy as np
import tensorflow as tf

def set_seeds(seed=42):
    """
    Sets the random seeds for deterministic results.

    Args:
        seed (int): The seed to use.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # The following line is needed for full determinism in TensorFlow >= 2.8
    # It might impact performance, but it's crucial for reproducibility.
    tf.config.experimental.enable_op_determinism()
    print(f"Random seeds set to {seed} for reproducibility.")

def plot_confusion_matrix(y_true, y_pred, labels=['Negative', 'Positive']):
    """
    Generates and displays a confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.show()

def plot_roc_curve(y_true, y_pred_proba, set_name=""):
    """
    Calculates the AUC score and plots the ROC curve.
    """
    auc_score = roc_auc_score(y_true, y_pred_proba)
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Random Guess')
    
    title_text = 'ROC Curve'
    if set_name:
        title_text += f' ({set_name})'
    
    plt.title(title_text, fontsize=16)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()