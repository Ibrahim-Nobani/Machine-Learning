import csv
import random
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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


print("pca")
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(current_time)
# steps = [('scaling', StandardScaler()), ('reduce_dim', PCA(n_components=50,tol=0.1)), ('clf', xgb.XGBClassifier(learning_rate=1, n_estimators=100))]
# steps = [('scaling', StandardScaler()), ('reduce_dim', PCA(n_components=85,tol=0.1)), ('clf', xgb.XGBClassifier(learning_rate=1, n_estimators=100))]
steps = [('scaling', StandardScaler()), ('reduce_dim', PCA(n_components=100,tol=0.1)), ('clf', xgb.XGBClassifier(learning_rate=1, n_estimators=100))]
pipeline = Pipeline(steps)

# train
pipeline.fit(x_train_split, y_train_split)

# predict
print("training finished")
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(current_time)
y_pred = pipeline.predict(x_train_split)
accuracy = accuracy_score(y_train_split, y_pred)
# print accuracy
print("Accuracy: {:.2f}%".format(accuracy * 100))
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(current_time)

y_pred = pipeline.predict(x_validation)
accuracy = accuracy_score(y_validation, y_pred)
# print accuracy
print("Accuracy: {:.2f}%".format(accuracy * 100))
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(current_time)

y_pred = pipeline.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
# print accuracy
print("Accuracy: {:.2f}%".format(accuracy * 100))
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(current_time)

for i in range(10):
    x = np.array(test_classes[i])
    tru = [i] * 1000
    tru = np.array(tru)
    predict_x = pipeline.predict(x)
    accuracy = accuracy_score(tru, predict_x)
    print("Accuracy for label " + str(i) + " : {:.2f}%".format(accuracy * 100))

predict = pipeline.predict(x_test)
con_matrix = pd.crosstab(
    pd.Series(y_test.ravel(), name="Actual"), pd.Series(predict, name="Predicted")
)

plt.title("Confusion Matrix")
sns.heatmap(con_matrix, cmap="Blues", annot=True, fmt="g")
plt.show()