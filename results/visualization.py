import matplotlib.pyplot as plt
import csv
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import numpy as np

def save_training_results_to_csv(all_loss, all_accuracy, all_epoch, filename="training_results.csv"):
    """
    Saves the training results to a CSV file.

    :param all_loss: List of loss values for each epoch.
    :param all_accuracy: List of accuracy values for each epoch.
    :param all_epoch: List of epoch numbers.
    :param filename: Name of the file to save the results.
    """
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Loss", "Accuracy"])  # Writing the headers
        for epoch, loss, accuracy in zip(all_epoch, all_loss, all_accuracy):
            writer.writerow([epoch, loss, accuracy])

    print(f"Training results saved to {filename}")


def save_test_results_to_csv(y_true, y_pred, filename="test_results.csv" ):
    """
    Saves the test results (true and predicted values) to a CSV file.

    :param y_true: List of true labels.
    :param y_pred: List of predicted labels.
    :param filename: Name of the file to save the results.
    """
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["True Label", "Predicted Label"])  # Writing the headers
        for true, pred in zip(y_true, y_pred):
            writer.writerow([true, pred])

    print(f"Test results saved to {filename}")


def plot_accuracy_for_networks(csv_files, title="Accuracy over Epochs for Multiple Networks", save_path="./results/accuracy_plot.png"):
    """
    Plots the accuracy over epochs for multiple neural networks on separate subplots for Task 1 and Task 2, with individual y-axis adjustments.

    :param csv_files: List of CSV file paths, each containing epoch, loss, and accuracy data for a network.
    :param title: Title of the plot.
    :param save_path: Path where the plot image will be saved.
    """
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=False)
    fig.suptitle(title)

    # Define colors and markers for models
    network_colors = {
        'nn': 'blue',   # Neural Network
        'gru': 'green', # GRU
        'rf': 'red',    # Random Forest
        'gbc': 'purple',# Gradient Boosting
        'cnn': 'orange' # CNN
    }
    network_markers = {
        'nn': 'o',   # Circle
        'gru': 's',  # Square
        'rf': '^',   # Triangle up
        'gbc': 'D',  # Diamond
        'cnn': 'p',  # Pentagon
    }

    # Initialize min and max accuracy values for Task 1
    min_accuracy_task1 = 100
    max_accuracy_task1 = 0

    for file in csv_files:
        data = pd.read_csv(file)
        if 'Epoch' in data.columns and 'Accuracy' in data.columns:
            # Extract model name and task number from the file name
            parts = os.path.splitext(os.path.basename(file))[0].split('_')
            model_name = parts[2].lower()
            task_number = parts[3]

            color = network_colors.get(model_name, 'black')
            marker = network_markers.get(model_name, '.')

            ax = axes[int(task_number) - 1]
            ax.plot(data['Epoch'], data['Accuracy'], label=f"{model_name.upper()}", color=color, marker=marker)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy (%)')
            ax.set_title(f"Task {task_number}")
            ax.legend()
            ax.grid(True)

            # Update min and max accuracy for Task 1 if applicable
            if task_number == '1':
                min_accuracy_task1 = min(min_accuracy_task1, data['Accuracy'].min())
                max_accuracy_task1 = max(max_accuracy_task1, data['Accuracy'].max())
        else:
            print(f"Warning: The file {file} does not contain 'Epoch' and 'Accuracy' columns.")

    # Set y-axis limits for Task 1 based on min and max accuracy values
    axes[0].set_ylim([min_accuracy_task1 - 5, max_accuracy_task1 + 5])

    plt.savefig(save_path)
    plt.close()

    print(f"Plot saved to {save_path}")


def plot_and_save_confusion_matrix(csv_files, title="Confusion Matrices", save_path="./results/confusion_matrices.png"):
    """
    Plots and saves a single plot containing all the confusion matrices for the specified test result CSV files,
    arranging one model per row and one task per column.

    :param csv_files: List of CSV file paths, each containing true and predicted labels.
    :param title: Title for the overall plot of confusion matrices.
    :param save_path: Path where the combined plot image will be saved.
    """
    # Extract unique models and tasks
    models = set()
    tasks = set()
    for file in csv_files:
        parts = os.path.basename(file).replace('test_result_', '').split('_')
        models.add(parts[0].lower())
        tasks.add(parts[1].split('.')[0])

    models = sorted(models)
    tasks = sorted(tasks, key=int)

    # Create mapping of models to rows and tasks to columns
    model_to_row = {model: i for i, model in enumerate(models)}
    task_to_col = {task: i for i, task in enumerate(tasks)}

    nrows, ncols = len(models), len(tasks)

    # Create a figure with subplots
    plt.figure(figsize=(ncols * 5, nrows * 4))
    plt.suptitle(title, fontsize=16)

    for file in csv_files:
        data = pd.read_csv(file)
        y_true = data['True Label']
        y_pred = data['Predicted Label']

        # Extract model name and task number
        parts = os.path.basename(file).replace('test_result_', '').split('_')
        model_name = parts[0].lower()
        task_number = parts[1].split('.')[0]

        # Determine the row and column for the current model and task
        row = model_to_row[model_name]
        col = task_to_col[task_number]

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Save confusion matrix data to CSV
        cm_df = pd.DataFrame(cm)
        cm_data_save_path = f"./results/results_csv/test/{model_name}_task{task_number}_raw_data.csv"
        cm_df.to_csv(cm_data_save_path, index=False)
        # Create subplot
        ax = plt.subplot(nrows, ncols, (row * ncols) + col + 1)
        sns.heatmap(cm, annot=False, fmt="d", square=True, cbar=False, ax=ax)  # Set square to True to ensure equal aspect ratio
        plt.title(f'{model_name.upper()} Task {task_number}')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the rect to make room for the suptitle

    # Save the combined plot
    plt.savefig(save_path)
    plt.close()

    print(f"Combined confusion matrix plot saved to {save_path}")
    print(f"Confusion matrix data saved to ./results/results_csv/test/")
