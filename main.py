from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from encryption import create_ckks_context
from learning_params import NUM_CLIENTS, NUM_ROUNDS, NUM_EPOCHS, BATCH_SIZE
from weights_util import encrypt_model_weights, decrypt_model_weights


# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


# Define a simple MLP model for MNIST
def create_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Simulate federated clients by splitting the training data
num_clients = NUM_CLIENTS
client_data_size = len(x_train) // num_clients
client_datasets = []
for i in range(num_clients):
    start = i * client_data_size
    end = start + client_data_size
    client_datasets.append((x_train[start:end], y_train[start:end]))


# Function to average weights from multiple models (HO - FedAvg)
def fed_avg(weights_list):
    avg_weights = []
    number_of_clients = len(weights_list)

    # Iterate through each layer's encrypted weights
    for layer_idx in range(len(weights_list[0])):
        # Get the encrypted weights for this layer from all clients
        layer_weights = [weights[layer_idx] for weights in weights_list]

        # Start with the first client's weights for this layer
        sum_weight = layer_weights[0]
        # someweight = someweight + layer_weights[i]
        # pyseal
        # Homomorphic addition of weights from other clients
        for index in range(1, number_of_clients):
            sum_weight = sum_weight + layer_weights[index]

        # Scale by 1/number_of_clients to get average using homomorphic multiplication
        avg_weight = sum_weight * (1.0 / number_of_clients)

        avg_weights.append(avg_weight)

    return avg_weights



# Initialize the global model
global_model = create_model()
global_weights_encrypted = None
global_weights_shapes = None
ckks_context = create_ckks_context()
# Federated training parameters
num_rounds = NUM_ROUNDS  # number of communication rounds
local_epochs = NUM_EPOCHS  # epochs of training on each client per round
batch_size = BATCH_SIZE
# Federated training simulation
for round_num in range(num_rounds):
    print(f"\n--- Federated Training Round {round_num + 1} ---")
    local_weights = []

    # Each client trains on its local data
    for client_index, (x_client, y_client) in enumerate(client_datasets):
        # Create a new local model
        local_model = create_model()

        if round_num == 0:
            # In round 0, use plaintext weights
            local_model.set_weights(global_model.get_weights())
        else:
            decrypted_weights = decrypt_model_weights(global_weights_encrypted, global_weights_shapes, ckks_context)
            local_model.set_weights(decrypted_weights)

        # Train the local model
        local_model.fit(x_client, y_client, epochs=local_epochs, batch_size=batch_size, verbose=0)

        client_weights = local_model.get_weights()
        encrypted_weights, original_shapes = encrypt_model_weights(client_weights, ckks_context)
        local_weights.append(encrypted_weights)

        global_weights_shapes = original_shapes

        print(f"Client {client_index + 1} done.")

    # Average the weights from all clients (FedAvg)
    global_weights_encrypted = fed_avg(local_weights)

    # For evaluation only: decrypt weights to update the global model
    decrypted_global_weights = decrypt_model_weights(global_weights_encrypted, global_weights_shapes, ckks_context)
    global_model.set_weights(decrypted_global_weights)

    # evaluation client

    # Evaluate the global model on the test data after each round
    loss, acc = global_model.evaluate(x_test, y_test, verbose=0)
    print(f"========= Round {round_num + 1} ==========")
    print(f"--------- Accuracy: {acc:.4f} ---------")
    print(f"--------- LOSS {loss} ---------")
# Final evaluation of the global model
loss, acc = global_model.evaluate(x_test, y_test, verbose=0)
print("\nFinal Test Accuracy:", acc)