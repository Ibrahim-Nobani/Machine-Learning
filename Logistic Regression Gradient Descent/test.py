import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Train Dataset
train = pd.read_csv("train.csv")
X = train[["x1", "x2"]].values
Y = train["class"].values
Y = list(map(encode_label, Y))
print(Y)
