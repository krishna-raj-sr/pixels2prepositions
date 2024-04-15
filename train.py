# Standard library imports
import numpy as np
import pandas as pd

# Third-party library imports
import tensorflow as tf
from tensorflow import keras

# Local imports
from keras_learning_rate_scheduler import LRSchedule
import constants as c
from helper import make_dataset, split_data
from model import get_model

# Dataset
dataset = pd.read_csv(c.PATH_TO_TRAINING_DATASET)
train_data, valid_data, test_data = split_data(dataset)
LEN_TRAIN_DATASET = len(train_data)

#Loss
categorical_cross_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits= False,reduction="none")

#Optimizer
adam = tf.keras.optimizers.Adam
num_train_steps = len(LEN_TRAIN_DATASET) * c.EPOCHS
num_warmup_steps = num_train_steps // 15
lr_schedule = LRSchedule(post_warmup_learning_rate=1e-4, warmup_steps=num_warmup_steps)
optimizer=adam(lr_schedule)

#Early Stopping Patience
early_stopping = keras.callbacks.EarlyStopping(patience=c.PATIENCE, restore_best_weights=True)

#Model
model =  get_model()
model.compile(optimizer, loss = categorical_cross_entropy, metrics=["accuracy"])

#Creating Dataset
train_dataset = make_dataset((train_data[c.HEADER_IMAGE_FILE_PATH]), train_data[c.HEADER_CLASS_VECTOR] , batch_size = c.BATCH_SIZE)
valid_dataset = make_dataset((valid_data[c.HEADER_IMAGE_FILE_PATH]), valid_data[c.HEADER_CLASS_VECTOR] , batch_size =c.BATCH_SIZE)
test_dataset  = make_dataset((test_data[c.HEADER_IMAGE_FILE_PATH]), test_data[c.HEADER_CLASS_VECTOR] , batch_size = c.BATCH_SIZE)

# Fit the model
history = model.fit(
            train_dataset,
            epochs=c.EPOCHS,
            batch_size=c.BATCH_SIZE,
            validation_data=valid_dataset,
            callbacks=[early_stopping],
  )

# Model Prediction
prediction = model.predict(test_dataset, verbose = False, batch_size = 50)