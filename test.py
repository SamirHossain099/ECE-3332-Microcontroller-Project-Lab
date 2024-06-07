import tensorflow as tf
from tensorflow_estimator.python.estimator.canned.linear import LinearRegressor

# Define feature columns
feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

# Create an estimator
estimator = LinearRegressor(feature_columns=feature_columns)

# Define an input function
def input_fn():
    return {"x": np.array([1., 2., 3., 4.])}, np.array([0., -1., -2., -3.])

# Train the estimator
estimator.train(input_fn=input_fn, steps=1000)

