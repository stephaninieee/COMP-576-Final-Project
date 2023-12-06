import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# make a new directory
def mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        print('> Creating: {}'.format(path))

# check for cuda and select gpu for training or otherwise cpu
def getDevice():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}, {torch.cuda.get_device_name(device)}')
    return device

# resample a target class if the dataset is heavily imbalanced
def rebalanceDataset(img_df, label_df, num_sample=10000):
    train_df = pd.concat([img_df, label_df], axis=1)
    train_df.columns = ['img', 'label']

    target_df = train_df[train_df['label'] == 0]
    resample_df = resample(target_df, replace=False, n_samples=num_sample)

    new_train_df = pd.concat([resample_df, train_df[train_df['label'] != 0]])

    return new_train_df['img'], new_train_df['label']

# evaluate the model by returning the predictions and target label
def evaluate(model, data_loader, device):
    predict = []
    target = []
    with torch.no_grad():
        model.eval()
        for idx, (x, label) in enumerate(data_loader):
            x, label = x.to(device), label.to(device)
            output = model(x)

            predict.extend(torch.argmax(output.data, 1).detach().cpu())
            target.extend(label.detach().cpu())
    return target, predict


def countplot(dataframe):
    ax = sns.countplot(x=dataframe)
    for num in ax.containers:
        ax.bar_label(num)
    plt.savefig('./figures/countplot.png')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes, normalize='true', title=None, cmap=plt.cm.Blues):
    matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize=normalize)
    ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=np.arange(classes)).plot(cmap=cmap)
    plt.title('Normalized confusion matrix ({})'.format(title))
    plt.savefig('./figures/confusion_matrix_' + title + '.png')
    # plt.show()

def show_train_result(num_epochs, test_loss, test_acc):
    x_range = np.arange(1, num_epochs+1)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Loss')
    for name in test_loss:
        plt.plot(x_range, test_loss[name], label=name)

    plt.subplot(1, 2, 2)
    plt.title('Accuracy')
    for name in test_acc:
        plt.plot(x_range, test_acc[name], label=name)
    plt.legend()
    plt.savefig('./figures/vgg_result.png')