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

x_t = x_train_split.reshape(-1, 28, 28, 1)
x_v = x_validation.reshape(-1, 28, 28, 1)
x_f = x_test.reshape(-1, 28, 28, 1)

# Convert the labels to categorical
y_t = keras.utils.to_categorical(y_train_split)
y_v = keras.utils.to_categorical(y_validation)
y_f = keras.utils.to_categorical(y_test)

# Define the model
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(current_time)
model = tf.keras.Sequential()
# Must define the input shape in the first layer of the neural network
model.add(
    tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
# Train the model
model.fit(x_t, y_t, epochs=10, batch_size=32, verbose=1)
# model.fit(x_t, y_t, epochs=10, batch_size=64, verbose=1)
# model.fit(x_t, y_t, epochs=18, batch_size=32, verbose=1)
# model.fit(x_t, y_t, epochs=18, batch_size=64, verbose=1)
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(current_time)
# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(x_t, y_t, verbose=1)
print("Test accuracy: {:.2f}%".format(test_acc * 100))
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(current_time)

test_loss, test_acc = model.evaluate(x_v, y_v, verbose=1)
print("Test accuracy: {:.2f}%".format(test_acc * 100))
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(current_time)

test_loss, test_acc = model.evaluate(x_f, y_f, verbose=1)
print("Test accuracy: {:.2f}%".format(test_acc * 100))
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(current_time)


for i in range(10):
    x = np.array(test_classes[i])
    x = x.reshape(-1, 28, 28, 1)
    tru = [i] * 1000
    tru = np.array(tru)
    predict_x = model.predict(x)
    classes_x = np.argmax(predict_x, axis=1)
    accuracy = accuracy_score(tru, classes_x)
    print("Accuracy for label " + str(i) + " : {:.2f}%".format(accuracy * 100))

predict = model.predict(x_f)
predict = np.argmax(predict, axis=1)
con_matrix = pd.crosstab(
    pd.Series(y_test.ravel(), name="Actual"), pd.Series(predict, name="Predicted")
)

plt.title("Confusion Matrix")
sns.heatmap(con_matrix, cmap="Blues", annot=True, fmt="g")
plt.show()
