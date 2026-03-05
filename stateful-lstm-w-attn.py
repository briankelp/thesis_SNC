import sys  # read command-line arguments (e.g., dataset path)
import pandas as pd  # load and manipulate tabular data (CSV) as DataFrames
import numpy as np  # numerical arrays + reshaping
import tensorflow as tf  # deep learning backend (not used directly, but imported)
from tensorflow import keras  # Keras high-level API
from tensorflow.keras import layers  # common neural network layers
from sklearn.model_selection import train_test_split  # split data into train/test sets
from sklearn.preprocessing import StandardScaler, LabelEncoder  # feature scaling + label encoding


def load_data(csv_file, label_column):
    data = pd.read_csv(csv_file)  # read CSV into a DataFrame (rows=samples, cols=features+label)
    features = data.drop(columns=[label_column])  # remove the label column -> feature matrix (X)
    labels = data[label_column]  # keep only the label column -> target vector (y)
    return features, labels  # return X and y


def preprocess_data(features, labels, seq_length):
    scaler = StandardScaler()  # create a scaler that standardizes each feature column
    features_scaled = scaler.fit_transform(features)  # fit on all features and transform to z-scores

    # If labels are strings/categories, convert them to integer IDs.
    if labels.dtype == 'object':
        encoder = LabelEncoder()  # maps unique labels to integers (e.g., {"benign":0,"mal":1})
        labels = encoder.fit_transform(labels)  # transform labels to integer array
    else:
        labels = labels.to_numpy()  # convert numeric pandas Series to a NumPy array

    labels = np.asarray(labels)  # ensure labels is a NumPy array (safe for later reshape)

    # We will build fixed-length sequences, so we may need to drop extra rows that don't fit.
    num_samples = len(features_scaled) // seq_length  # number of full sequences available
    truncate_len = num_samples * seq_length  # total rows that fit perfectly into sequences

    # Keep only the rows that fit and reshape into (num_sequences, seq_length, num_features).
    features_reshaped = features_scaled[:truncate_len].reshape(num_samples, seq_length, -1)

    # Do the same reshape for labels so each sequence has seq_length labels.
    labels_reshaped = labels[:truncate_len].reshape(num_samples, seq_length)

    # Use only the last label in each sequence as the sequence-level target.
    labels_final = labels_reshaped[:, -1].reshape(-1, 1)  # shape (num_sequences, 1)

    # Print basic shape/debug info so you can see what truncation/sequence building did.
    print(f"Original data length: {len(features_scaled)}")  # original row count
    print(f"Truncated data length: {truncate_len}")  # rows used after truncation
    print(f"Number of sequences: {num_samples}")  # number of sequences created

    return features_reshaped, labels_final  # return X as sequences, y as per-sequence labels


def create_model(input_shape, lstm_units, seq_length):
    inputs = keras.Input(shape=input_shape)  # model input: (seq_length, num_features)

    # LSTM processes the sequence and outputs a hidden state at each time step.
    H = layers.LSTM(lstm_units, return_sequences=True)(inputs)  # H: (batch, seq, units)

    # Self-attention: each time step attends to all time steps (query=H, value=H).
    context_seq = layers.Attention()([H, H])  # context_seq: (batch, seq, units)

    # Pool across time to get a single vector representing the attended sequence.
    context_vec = layers.GlobalAveragePooling1D()(context_seq)  # (batch, units)

    # Extract the last LSTM hidden state (last time step) as another summary vector.
    h_t = layers.Lambda(lambda x: x[:, -1, :])(H)  # (batch, units)

    # Combine the attention summary and the last hidden state.
    combined = layers.Concatenate(axis=-1)([context_vec, h_t])  # (batch, 2*units)

    # Project combined representation back to lstm_units with tanh activation.
    h_t_prime = layers.Dense(lstm_units, activation="tanh")(combined)  # (batch, units)

    # Final prediction for binary classification (probability in [0,1]).
    outputs = layers.Dense(1, activation="sigmoid")(h_t_prime)  # (batch, 1)

    model = keras.Model(inputs, outputs)  # assemble the Keras model
    return model  # return uncompiled model


def main(
    csv_file,
    label_column,
    seq_length=10,
    lstm_units=128,
    test_size=0.2,
    random_state=42,
    batch_size=32
):
    # Load raw tabular data from disk.
    features, labels = load_data(csv_file, label_column)

    # Scale features + convert into fixed-length sequences; make one label per sequence.
    features_reshaped, labels_final = preprocess_data(features, labels, seq_length)

    # Split sequences into training and test sets (random shuffle split).
    X_train, X_test, y_train, y_test = train_test_split(
        features_reshaped,  # X: (num_sequences, seq_length, num_features)
        labels_final,       # y: (num_sequences, 1)
        test_size=test_size,         # fraction of sequences in test set
        random_state=random_state    # fixed seed for reproducibility
    )

    # Ensure labels are float32 for TensorFlow and have shape (N,1).
    y_train = y_train.astype(np.float32).reshape(-1, 1)
    y_test = y_test.astype(np.float32).reshape(-1, 1)

    # Define the model input shape: (seq_length, num_features).
    input_shape = (seq_length, X_train.shape[2])

    # Build model architecture.
    model = create_model(input_shape, lstm_units, seq_length)

    # Configure training: optimizer, loss for binary classification, and accuracy metric.
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Train model on training set; validate on test set each epoch.
    model.fit(
        X_train, y_train,                 # training data
        epochs=10,                        # number of passes through the training set
        batch_size=batch_size,            # sequences per gradient update
        validation_data=(X_test, y_test)  # evaluate on held-out test set during training
    )

    # Final evaluation on test set after training.
    loss, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size)

    # Print final test performance.
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")


if __name__ == "__main__":
    # This block runs only when you execute the script directly (not when imported).
    if len(sys.argv) < 2:  # ensure a dataset path argument was provided
        print(f"Usage: {sys.argv[0]} <[fbs_nas/fbs_rrc.csv]>")  # show expected usage
        sys.exit(1)  # exit with error code

    dataset = sys.argv[1]  # first CLI arg is the CSV file path
    main(dataset, "label")  # run pipeline assuming the label column is named "label"