#select venv interpreter
import numpy as np
import os
from tflite_model_maker.config import ExportFormat, QuantizationConfig
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
from tflite_support import metadata

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

#load data
train_data = object_detector.DataLoader.from_pascal_voc(
    'TEST_IMGS/android_figurine/train',
    'TEST_IMGS/android_figurine/train',
    ['android','pig_android']
)
val_data = object_detector.DataLoader.from_pascal_voc(
    'TEST_IMGS/android_figurine/validate',
    'TEST_IMGS/android_figurine/validate',
    ['android','pig_android']
)

print(f"Size of train_data: {len(train_data)}")
print(f"Size of val_data: {len(val_data)}")
#choose model architecture
spec = model_spec.get('efficientdet_lite0')
model = object_detector.create(train_data, model_spec=spec, batch_size=4, train_whole_model=True, epochs=20, validation_data=val_data)
model.evaluate(val_data)