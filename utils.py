import random
from PIL import ImageFilter
import numpy as np
import torch
import string

def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str
    
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def shot_acc(preds, labels, train_data, many_shot_thr=1000, low_shot_thr=200, acc_per_cls=False):
    """
    It takes in the predictions and labels of a model, the training data, and thresholds for what
    constitutes a "many-shot" class, a "low-shot" class, and a "median-shot" class. It then returns the
    accuracy of the model on the many-shot, median-shot, and low-shot classes
    
    :param preds: the predictions of the model
    :param labels: the labels of the test set
    :param train_data: the training data
    :param many_shot_thr: the number of training samples per class that is considered "many shot",
    defaults to 100 (optional)
    :param low_shot_thr: the number of training samples per class below which we consider a class to be
    low-shot, defaults to 20 (optional)
    :param acc_per_cls: If True, returns a list of accuracies for each class, defaults to False
    (optional)
    :return: The mean of the many shot, median shot, and low shot accuracies.
    """
    if isinstance(train_data, np.ndarray):
        training_labels = np.array(train_data).astype(int)
    else:
        training_labels = np.array(train_data.dataset.labels).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError('Type ({}) of preds not supported'.format(type(preds)))
    train_class_count = []
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        class_correct.append((preds[labels == l] == labels[labels == l]).sum())

    many_shot = []
    median_shot = []
    low_shot = []

    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_class_count[i] < low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))
    print('many shot count: ', len(many_shot))
    print('median shot count: ', len(median_shot))
    print('low shot count: ', len(low_shot))
    if len(many_shot) == 0:
        many_shot.append(0)
    if len(median_shot) == 0:
        median_shot.append(0)
    if len(low_shot) == 0:
        low_shot.append(0)

    if acc_per_cls:
        class_accs = [c / cnt for c, cnt in zip(class_correct, test_class_count)]
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot), class_accs
    else:
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)
