# DATASET INFO
# This dataset uses the kaggle digit recognizer dataset(in CSV format) from here and reformats it so that there are 42,000 files. 
# Each with 784 greyscale values(28 per row, 28 columns) with the last value being the expected output value(what the number actually is).
# So 42k images of 28x28 and 784 pixels each 
# (https://www.kaggle.com/competitions/digit-recognizer/data?select=train.csv)
import pandas as pd
import numpy as np

X = pd.read_csv("/Users/sarthakkapila/Downloads/train.csv")

X = np.array(X)
np.random.shuffle(X)
labels = X[0]
inputs = X[1:786]

print(X.shape, X.head, labels.shape, inputs.shape)

# Spliting dataset 80/20
X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)