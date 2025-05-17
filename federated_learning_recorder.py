import os
import pandas as pd
import matplotlib.pyplot as plt

from learning_params import NUM_ROUNDS, NUM_CLIENTS, METHODS


class FederatedLearningRecorder:

    def __init__(self, num_clients=NUM_CLIENTS, num_rounds=NUM_ROUNDS, method=METHODS[1]):
        """
        Initialize the recorder with a base directory for saving results.
        """
        self.base_dir = 'federated_learning_results'+f'/{method}'
        os.makedirs(self.base_dir, exist_ok=True)
        self.rounds = []
        self.losses = []
        self.accuracies = []
        self.best_accuracy = 0.0
        self.best_round = 0
        self.num_clients = num_clients
        self.num_rounds = num_rounds

    def add_round_metrics(self, round_num, loss, accuracy):
        """
        Add metrics for a specific round.
        """
        self.rounds.append(round_num)
        self.losses.append(loss)
        self.accuracies.append(accuracy)


        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_round = round_num

    def save_fl_run_to_csv(self):
        """
        Save the metrics to a CSV file.
        """
        result_dir = os.path.join(
            self.base_dir,
            f"{self.num_rounds}-rounds",
            f"{self.num_clients}-clients"
        )
        os.makedirs(result_dir, exist_ok=True)

        filepath = os.path.join(
            result_dir,
            f"fl_run_metrics_num_clients-{self.num_clients}.csv"
        )

        data_df = pd.DataFrame(
            {'round': self.rounds,
             'loss': self.losses,
             'accuracy': self.accuracies,
             'num_clients': self.num_clients}
        )
        data_df.to_csv(filepath, index=False)
        print(f"Federated learning run metrics saved to {filepath}")
        print(f"Best accuracy: {self.best_accuracy} at round {self.best_round}")
