import csv
import random
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn import pipeline

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
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



num_instances = x_train_split.shape[0]

# Number of elements in each chunk (28 in your case)
chunk_size = 28

# Compute the average for each chunk and replace the chunk with the average
for i in range(num_instances):
    for j in range(0, 784, chunk_size):
        chunk = x_train_split[i, j:j+chunk_size]
        avg = np.mean(chunk)
        x_train_split[i, j:j+chunk_size] = avg

num_instances = x_validation.shape[0]

# Number of elements in each chunk (28 in your case)
chunk_size = 28

# Compute the average for each chunk and replace the chunk with the average
for i in range(num_instances):
    for j in range(0, 784, chunk_size):
        chunk = x_validation[i, j:j+chunk_size]
        avg = np.mean(chunk)
        x_validation[i, j:j+chunk_size] = avg

num_instances = x_test.shape[0]

# Number of elements in each chunk (28 in your case)
chunk_size = 28

# Compute the average for each chunk and replace the chunk with the average
for i in range(num_instances):
    for j in range(0, 784, chunk_size):
        chunk = x_test[i, j:j+chunk_size]
        avg = np.mean(chunk)
        x_test[i, j:j+chunk_size] = avg

def train_random_forest(xTrain, yTrain, xValid, yValid, xTest, yTest, n_estimators, criterion):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(current_time)
    clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion)
    # clf = RandomForestClassifier()
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

train_random_forest(x_train_split, y_train_split, x_validation, y_validation, x_test, y_test, 500,'entropy')



