import os
import pandas as pd
import matplotlib.pyplot as plt

from learning_params import NUM_ROUNDS, NUM_CLIENTS


def plot_fl_results(rounds, clients):
    """
    Plot federated learning results from a CSV file.

    Args:
        csv_path: Path to the CSV file containing federated learning metrics
        output_dir: Directory to save the plots
        loss_fname: Filename for the loss plot
        acc_fname: Filename for the accuracy plot
    """
    base_dir = 'federated_learning_results'
    result_dir = os.path.join(base_dir, f"{rounds}-rounds", f"{clients}-clients")
    csv_path = os.path.join(result_dir, f"fl_run_metrics_num_clients-{clients}.csv")

    # Read the CSV file
    df = pd.read_csv(csv_path)
    loss_fname = f'fl_loss_clients-{clients}.png'
    acc_fname = f'fl_accuracy_clients-{clients}.png'
    # Get the number of clients
    num_clients = df['num_clients'].iloc[0] if 'num_clients' in df.columns else "Unknown"
    # Save plots in the same directory as the CSV file
    plots_dir = result_dir  # Using the same directory for simplicity
    os.makedirs(plots_dir, exist_ok=True)
    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(df['round'], df['loss'], marker='o')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.title(f'Global Loss over Rounds (Clients: {num_clients})')
    plt.grid(True)
    loss_path = os.path.join(result_dir, loss_fname)
    plt.savefig(loss_path)
    print(f"Saved loss plot to {loss_path}")
    plt.close()

    # Plot accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(df['round'], df['accuracy'], marker='o', color='green')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title(f'NUM ROUNDS - {rounds} - Global Model Accuracy (Clients: {num_clients})')
    plt.grid(True)
    acc_path = os.path.join(plots_dir, acc_fname)
    plt.savefig(acc_path)
    print(f"Saved accuracy plot to {acc_path}")
    plt.close()


if __name__ == "__main__":
    rounds = 25
    clients = 10
    plot_fl_results(clients=clients, rounds=rounds)