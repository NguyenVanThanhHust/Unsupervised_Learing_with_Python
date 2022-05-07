import os
from os.path import join, isdir, isfile
import pandas as pd
import numpy as np

from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split 
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc, roc_auc_score

DATA_PATH = "../../creditcard.csv"
data = pd.read_csv(DATA_PATH)

print(data.head())

print(data.describe())

dataX = data.copy().drop(["Class"], axis=1)
dataY = data["Class"].copy()

X_train, y_train, X_test, y_test = train_test_split.(dataX, dataY, test_size=0.33, \
                                        random_state=2018, stratify=dataY)

def anomalyScores(originalDF, reducedDF):
    """
    calcualte anomaly scores, that is diff
    """
    loss = np.sum(np.array(originalDF) - np.array(reducedDF)**2, axis=1)
    loss = pd.Series(data=loss, index=originalDF.index)
    loss = (loss - np.min(loss))/(np.max(loss) - np.min(loss))
    return loss

# Plot results
def plotResults(trueLabels, anomalyScores, returnPreds = False):
    preds = pd.concat([trueLabels, anomalyScores], axis=1)
    preds.columns = ['trueLabel', 'anomalyScore']
    precision, recall, thresholds = \
        precision_recall_curve(preds['trueLabel'],preds['anomalyScore'])
    average_precision = \
        average_precision_score(preds['trueLabel'],preds['anomalyScore'])

    plt.step(recall, precision, color='k', alpha=0.7, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.3, color='k')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])

    plt.title('Precision-Recall curve: Average Precision = \
    {0:0.2f}'.format(average_precision))

    fpr, tpr, thresholds = roc_curve(preds['trueLabel'], \
                                     preds['anomalyScore'])
    areaUnderROC = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='r', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic: \
    Area under the curve = {0:0.2f}'.format(areaUnderROC))
    plt.legend(loc="lower right")
    plt.savefig("result.jpg")

    if returnPreds==True:
        return preds

print(isfile(DATA_PATH))
