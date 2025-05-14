# ai_model.py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class AIModel:
    def __init__(self):
        self.model = None
        self._build_model()

    def _build_model(self):
        # Example neural network architecture
        self.model = keras.Sequential([
            layers.InputLayer(input_shape=(784,)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(10, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def get_model(self):
        return self.model

    def train(self, train_data, train_labels, epochs=5):
        self.model.fit(train_data, train_labels, epochs=epochs)

    def evaluate(self, test_data, test_labels):
        return self.model.evaluate(test_data, test_labels)
