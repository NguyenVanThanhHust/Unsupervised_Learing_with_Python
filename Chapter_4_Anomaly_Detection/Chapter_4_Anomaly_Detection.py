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

print(isfile(DATA_PATH))
