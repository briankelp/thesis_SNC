import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Attention, Concatenate, Input
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(csv_file, label_column):
    data = pd.read_csv(csv_file)
    features = data.drop(columns=[label_column])
    labels = data[label_column]
    return features, labels

def preprocess_data(features, labels, seq_length):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # If labels are categorical (object dtype), we encode them first
    if labels.dtype == 'object':
        encoder = LabelEncoder()
        labels = encoder.fit_transform(labels)
    else:
        # Otherwise, ensure labels is a NumPy array
        labels = labels.to_numpy()

    # Convert to NumPy array just to be certain
    labels = np.asarray(labels)

    # Number of full sequences we can form
    num_samples = len(features_scaled) // seq_length
    
    # Truncate the data to be divisible by seq_length
    truncate_len = num_samples * seq_length
    
    # Reshape features
    features_reshaped = features_scaled[:truncate_len]
    features_reshaped = features_reshaped.reshape(num_samples, seq_length, -1)

    # Reshape labels - using the same truncated length
    labels_reshaped = labels[:truncate_len]
    labels_reshaped = labels_reshaped.reshape(num_samples, seq_length)

    # Take the last label in each sequence as the target
    labels_final = labels_reshaped[:, -1].reshape(-1, 1)  # Ensures shape (num_samples, 1)


    print(f"Original data length: {len(features_scaled)}")
    print(f"Truncated data length: {truncate_len}")
    print(f"Number of sequences: {num_samples}")

    return features_reshaped, labels_final


class StatefulLSTM:
    def __init__(self, units, seq_length):
        self.units = units
        self.seq_length = seq_length
        # Set batch_input_shape to None to allow dynamic batch sizes
        self.lstm = LSTM(units, return_sequences=True, 
                        stateful=False)  # Changed stateful to False

    def __call__(self, inputs):
        outputs = self.lstm(inputs)
        return outputs

class LSTMwithAttention:
    def __init__(self, units):
        self.units = units
        self.lstm = LSTM(units, return_sequences=True)
        self.attention = Attention()
        self.concat = Concatenate(axis=-1)  # Specify axis for concatenation
        self.dense = Dense(units, activation='tanh')

    def __call__(self, inputs):
        H = self.lstm(inputs)  # Shape: (batch_size, seq_length, units)
        context_vector = self.attention([H, H])  # Shape: (batch_size, seq_length, units)
        # Get the last hidden state
        h_t = H[:, -1, :]  # Shape: (batch_size, units)
        # Reduce context vector to match h_t shape
        context_vector = tf.reduce_mean(context_vector, axis=1)  # Shape: (batch_size, units)
        # Concatenate along the feature axis
        combined = self.concat([context_vector, h_t])
        h_t_prime = self.dense(combined)
        return h_t_prime

def create_model(input_shape, lstm_units, seq_length):
    inputs = Input(shape=input_shape)
    stateful_lstm = StatefulLSTM(lstm_units, seq_length)
    h_t = stateful_lstm(inputs)
    lstm_attention = LSTMwithAttention(lstm_units)
    h_t_prime = lstm_attention(h_t)
    outputs = Dense(1, activation='sigmoid')(h_t_prime)  # Binary classification
    model = Model(inputs, outputs)
    return model

def main(csv_file, label_column, seq_length=10, lstm_units=128, test_size=0.2, random_state=42, batch_size=32):
    # Load data
    features, labels = load_data(csv_file, label_column)

    # Preprocess data
    features_reshaped, labels_final = preprocess_data(features, labels, seq_length)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features_reshaped, labels_final, test_size=test_size, random_state=random_state)

    y_train = y_train.astype(np.float32).reshape(-1, 1)
    y_test = y_test.astype(np.float32).reshape(-1, 1)

    # Create the model
    input_shape = (seq_length, X_train.shape[2])
    model = create_model(input_shape, lstm_units, seq_length)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=batch_size, validation_data=(X_test, y_test))

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <[fbs_nas/fbs_rrc.csv]>")
        sys.exit(1)
    dataset = sys.argv[1]

    main(dataset, "label")