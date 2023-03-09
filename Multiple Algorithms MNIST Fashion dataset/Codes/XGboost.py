import csv
import random
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_fscore_support
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix
import checkpointer as checkpointer
import tensorflow as tf
import seaborn as sns
from tensorflow import keras
from keras.optimizers import Adadelta, RMSprop, Adam
from sklearn.model_selection import train_test_split
import csv
import random
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_fscore_support
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix
import checkpointer as checkpointer
import tensorflow as tf
import seaborn as sns
from tensorflow import keras
from keras.optimizers import Adadelta, RMSprop, Adam
from sklearn.model_selection import train_test_split
import mnist_reader
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import numpy as np
from datetime import datetime
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, MaxPool2D, Dropout
import threading
from sklearn.ensemble import ExtraTreesClassifier
import xgboost as xgb

x_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
x_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
r = random.randrange(0, 50)
x_train_shuffled, y_train_shuffled = shuffle(x_train, y_train, random_state=r)
x_train_split, x_validation, y_train_split, y_validation = train_test_split(x_train_shuffled, y_train_shuffled,
                                                                            test_size=2 / 6, random_state=r)

x_train_split = x_train_split / 255
x_validation = x_validation / 255
x_test = x_test / 255

test_classes = [[] for i in range(10)]
counter = 0
for i in y_test:
    test_classes[i].append(x_test[counter])
    counter += 1

def train_xgb(xTrain, yTrain, xValid, yValid, xTest, yTest, learning_rate, n_estimators):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(current_time)
    clf = xgb.XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators)
    clf.fit(xTrain, yTrain)

    print("training finished")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(current_time)
    y_pred = clf.predict(xTrain)
    accuracy = accuracy_score(yTrain, y_pred)
    # print accuracy
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(current_time)

    y_pred = clf.predict(xValid)
    accuracy = accuracy_score(yValid, y_pred)
    # print accuracy
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(current_time)

    y_pred = clf.predict(xTest)
    accuracy = accuracy_score(yTest, y_pred)
    # print accuracy
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(current_time)
    print(classification_report(yTest, y_pred))
    for i in range(10):
        tru = [i] * 1000
        tru = np.array(tru)
        predict_x = clf.predict(test_classes[i])
        accuracy = accuracy_score(tru, predict_x)
        print("Accuracy for label " + str(i) + " : {:.2f}%".format(accuracy * 100))

    predict = clf.predict(x_test)
    con_matrix = pd.crosstab(
        pd.Series(y_test.ravel(), name="Actual"), pd.Series(predict, name="Predicted")
    )

    plt.title("Confusion Matrix")
    sns.heatmap(con_matrix, cmap="Blues", annot=True, fmt="g")
    plt.show()

train_xgb(x_train_split, y_train_split, x_validation, y_validation, x_test, y_test, learning_rate=0.1, n_estimators=50)
# train_xgb(x_train_split, y_train_split, x_validation, y_validation, x_test, y_test, learning_rate=0.01, n_estimators=50)
# train_xgb(x_train_split, y_train_split, x_validation, y_validation, x_test, y_test, learning_rate=0.1, n_estimators=100)
# train_xgb(x_train_split, y_train_split, x_validation, y_validation, x_test, y_test, learning_rate=0.01, n_estimators=100)