# src/model.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_lstm_model(input_shape, lstm_units=100, dropout_rate=0.3, learning_rate=0.001):
    """
    Builds a tunable LSTM model architecture.
    This function allows for easy hyperparameter testing.

    Args:
        input_shape (tuple): The shape of the input data (time_steps, n_features).
        lstm_units (int): Number of units in the first LSTM layer.
        dropout_rate (float): Dropout rate for regularization.
        learning_rate (float): Learning rate for the Adam optimizer.

    Returns:
        A compiled Keras model.
    """
    model = Sequential([
        tf.keras.Input(shape=input_shape),
        LSTM(units=lstm_units, return_sequences=True), # return_sequences=True for stacking LSTM Layers
        Dropout(dropout_rate),
        LSTM(units=int(lstm_units / 2)), # Second LSTM Layer with half the units
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid') # Sigmoid activation for binary classification
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer, 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    
    return model