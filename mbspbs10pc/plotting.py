import itertools

import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return plt


def plot_history(history):
    """Plot training loss and accuracy."""
    plt.figure(dpi=100)
    t = history.epoch

    plt.subplot(211)
    plt.plot(t, history.history['loss'], label='loss', color='C0')
    plt.plot(t, history.history['val_loss'], label='val_loss', color='C1')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(loc=1)

    plt.subplot(212)
    plt.plot(t, history.history['acc'], label='acc', color='C2')
    plt.plot(t, history.history['val_acc'], label='val_acc', color='C3')
    plt.ylim([0.5, 1])
    plt.ylabel('acc')
    plt.xlabel('epochs')
    plt.legend(loc=1)

    plt.tight_layout()
    return plt
