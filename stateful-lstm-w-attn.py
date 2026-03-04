import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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

    if labels.dtype == 'object':
        encoder = LabelEncoder()
        labels = encoder.fit_transform(labels)
    else:
        labels = labels.to_numpy()

    labels = np.asarray(labels)

    num_samples = len(features_scaled) // seq_length
    truncate_len = num_samples * seq_length

    features_reshaped = features_scaled[:truncate_len].reshape(num_samples, seq_length, -1)

    labels_reshaped = labels[:truncate_len].reshape(num_samples, seq_length)
    labels_final = labels_reshaped[:, -1].reshape(-1, 1)

    print(f"Original data length: {len(features_scaled)}")
    print(f"Truncated data length: {truncate_len}")
    print(f"Number of sequences: {num_samples}")

    return features_reshaped, labels_final

def create_model(input_shape, lstm_units, seq_length):
    inputs = keras.Input(shape=input_shape)

    # LSTM over the sequence
    H = layers.LSTM(lstm_units, return_sequences=True)(inputs)  # (batch, seq, units)

    # Self-attention over time
    context_seq = layers.Attention()([H, H])  # (batch, seq, units)

    # Reduce time dimension using a Keras layer (instead of tf.reduce_mean)
    context_vec = layers.GlobalAveragePooling1D()(context_seq)  # (batch, units)

    # Last hidden state (use a Lambda layer to keep it in Keras land)
    h_t = layers.Lambda(lambda x: x[:, -1, :])(H)  # (batch, units)

    combined = layers.Concatenate(axis=-1)([context_vec, h_t])
    h_t_prime = layers.Dense(lstm_units, activation="tanh")(combined)

    outputs = layers.Dense(1, activation="sigmoid")(h_t_prime)  # binary classification
    model = keras.Model(inputs, outputs)
    return model

def main(csv_file, label_column, seq_length=10, lstm_units=128, test_size=0.2, random_state=42, batch_size=32):
    features, labels = load_data(csv_file, label_column)
    features_reshaped, labels_final = preprocess_data(features, labels, seq_length)

    X_train, X_test, y_train, y_test = train_test_split(
        features_reshaped, labels_final, test_size=test_size, random_state=random_state
    )

    y_train = y_train.astype(np.float32).reshape(-1, 1)
    y_test = y_test.astype(np.float32).reshape(-1, 1)

    input_shape = (seq_length, X_train.shape[2])
    model = create_model(input_shape, lstm_units, seq_length)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(X_train, y_train, epochs=10, batch_size=batch_size, validation_data=(X_test, y_test))

    loss, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <[fbs_nas/fbs_rrc.csv]>")
        sys.exit(1)
    dataset = sys.argv[1]
    main(dataset, "label")