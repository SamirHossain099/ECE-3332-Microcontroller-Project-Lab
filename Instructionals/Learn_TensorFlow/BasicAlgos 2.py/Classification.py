"""
Classification
Regression is used to predict a numeric value, classification is used to seperate data points
into classes of different labels. IN this example we will use a TensorFlow estimator to classify flowers.

"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd

"""
Dataset info:
This specific dataset seperates flowers into 3 different classes of species:
Setosa, Versicolor, Virginica

The information about each flower is the following:
Sepal length, sepal width, petal length, petal width

"""

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLenght', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
#Defining column names
train_path = tf.estimator.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.estimator.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
# Here we use keras (a module inside of TensorFlow) to grab our datasets and read them into a pandas dataframe

#print(train.head())
#notice the species are already encoded for us.
train_y = train.pop('Species')
test_y = test.pop('Species')
print(train.head()) #the species column is now gone

#input function
def input_fn(features, labels, training=True, batch_size=256):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()
    
    return dataset.batch(batch_size)

# Feature columns describe how to use the input.
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
print(my_feature_columns)

#Building the model
#DNN Classifier
# Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each.
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 30 and 10 nodes respectively.
    hidden_units=[30, 10],
    # The model must choose between 3 classes.
    n_classes=3)