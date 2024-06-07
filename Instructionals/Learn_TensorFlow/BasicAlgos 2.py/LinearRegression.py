"""
Linear Regression Notes:

One of the most basic forms of machine learning to find linear relations between datasets
The algorithm uses the line of best fit for the data points to predict other points.
With linear regression you can predict a data point as long as you have n-1 of n dimensions.

Steps: 
1. Import
2. Load data set
3. Explore data set
4. Create catagorical and numeric columns
5. Encode the catagorical and numeric columns
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow as tf
from tensorflow.keras import layers

## Titanic Example
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')  # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')  # testing data
y_train = dftrain.pop('survived')  # remove label we are training to predict
y_eval = dfeval.pop('survived')

# we need to encode the categorical data with numbers.
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

# Convert data to tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((dict(dftrain), y_train))
eval_dataset = tf.data.Dataset.from_tensor_slices((dict(dfeval), y_eval))

# Feature input layers
feature_inputs = {}
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()
    feature_inputs[feature_name] = layers.Input(name=feature_name, shape=(), dtype=tf.string)
    feature_inputs[feature_name + '_encoded'] = layers.StringLookup(vocabulary=vocabulary)(feature_inputs[feature_name])

for feature_name in NUMERIC_COLUMNS:
    feature_inputs[feature_name] = layers.Input(name=feature_name, shape=(), dtype=tf.float32)

# Build the model
x = tf.keras.layers.Concatenate()(list(feature_inputs.values())[1:])
x = layers.Dense(32, activation='relu')(x)
x = layers.Dense(16, activation='relu')(x)
output = layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=list(feature_inputs.values()), outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())

# Train the model
model.fit(train_dataset.batch(32), epochs=10)

# Evaluate the model
result = model.evaluate(eval_dataset.batch(32), return_dict=True)
clear_output()
print(result['accuracy'])

# Make predictions
predictions = model.predict(eval_dataset.batch(32))
print(dfeval.loc[3])
print(y_eval.loc[3])
print(predictions[3][0])