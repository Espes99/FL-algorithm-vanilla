import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

from federated_learning_recorder import FederatedLearningRecorder
from learning_params import NUM_ROUNDS, NUM_EPOCHS, NUM_CLIENTS, BATCH_SIZE, METHODS
from plain_mlp_client.mlp_model import MLPModel

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Simulate federated clients by splitting the training data
num_clients = NUM_CLIENTS
client_data_size = len(x_train) // num_clients
client_datasets = []
for i in range(num_clients):
    start = i * client_data_size
    end = start + client_data_size
    client_datasets.append((x_train[start:end], y_train[start:end]))


# Function to average weights from multiple models (FedAvg)
def fed_avg(weights_list):
    avg_weights = list()
    # zip gathers the corresponding weight arrays from each client model
    for weights in zip(*weights_list):
        avg_weights.append(np.mean(weights, axis=0))
    return avg_weights

fl_recorder = FederatedLearningRecorder(num_clients=NUM_CLIENTS, method=METHODS[0])

# Initialize the global model
global_model = MLPModel()

# Federated training parameters
num_rounds = NUM_ROUNDS
local_epochs = NUM_EPOCHS  # epochs of training on each client per round
batch_size = BATCH_SIZE
curr_best_acc, best_round = 0, 0
# Federated training simulation
for round_num in range(num_rounds):
    print(f"\n--- Federated Training Round {round_num + 1} ---")
    local_weights = []

    # Each client trains on its local data
    for client_index, (x_client, y_client) in enumerate(client_datasets):
        # Create a new local model and set it to the current global weights
        local_model = MLPModel()
        local_model.model.set_weights(global_model.model.get_weights())

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='accuracy', patience=5, min_delta=0.005, mode='max'
        )

        # Train the local model
        local_model.model.fit(x_client, y_client, epochs=local_epochs, batch_size=batch_size, callbacks=[early_stopping], verbose=0)

        # Collect the updated weights from this client
        local_weights.append(local_model.model.get_weights())
        print(f"Client {client_index + 1} done.")

    # Average the weights from all clients (FedAvg)
    new_weights = fed_avg(local_weights)
    global_model.model.set_weights(new_weights)

    # Evaluate the global model on the test data after each round
    loss, acc = global_model.evaluate(x_test, y_test, verbose=0)
    fl_recorder.add_round_metrics(round_num=round_num+1, loss=loss, accuracy=acc)
    if acc > curr_best_acc:
        curr_best_acc, best_round = acc, round_num+1
        print(f"New Best Accuracy: {curr_best_acc:.4f} at round {best_round}")
    print(f"========= Round {round_num + 1} ==========")
    print(f"--------- Accuracy: {acc:.4f} ---------")
    print(f"--------- LOSS {loss} ---------")

# Final evaluation of the global model
loss, acc = global_model.evaluate(x_test, y_test, verbose=0)
print("\nFinal Test Accuracy:", acc)
print(f"Final Best Accuracy: {curr_best_acc} in round {best_round}")
fl_recorder.save_fl_run_to_csv()
