import torch
import torch.nn as nn
import torch.optim as optim
from skorch.callbacks import EarlyStopping
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


class GRUNeuralNetwork(nn.Module):
    """
    A GRU (Gated Recurrent Unit) Neural Network for sequence modeling.

    Attributes:
        gru (nn.GRU): The GRU layer.
        activation (Callable): Activation function.
        layer2 (nn.Linear): A linear layer for output.

    Args:
        output_size (int): The size of the output layer.
        input_size (int, optional): The size of the input layer. Defaults to 102.
        hidden_size1 (int, optional): The size of the hidden layer in the GRU. Defaults to 64.
        activation (str, optional): Type of activation function to use ('relu', 'sigmoid', 'tanh'). Defaults to 'relu'.
    """

    def __init__(self, output_size, input_size=102, hidden_size1=64, activation='relu'):
        super(GRUNeuralNetwork, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size1)
        self.activation = self.get_activation(activation)
        self.layer2 = nn.Linear(hidden_size1, output_size)

    def forward(self, x):
        """
        Forward pass of the neural network.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            Tensor: The output of the network.
        """
        output, _ = self.gru(x)
        output = self.activation(output)
        output = self.layer2(output)
        return output

    def get_activation(self, activation):
        """
        Retrieves the activation function based on the given string.

        Args:
            activation (str): The name of the activation function.

        Returns:
            Callable: The corresponding activation function.

        Raises:
            ValueError: If the activation function is not supported.
        """
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")


def restart_gru_model(learning_rate=0.0001):
    """
    Restarts and initializes the GRU model with the specified learning rate.

    Args:
        learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.0001.

    Returns:
        GRUNeuralNetwork: The initialized GRU neural network.
    """
    modelGRU = GRUNeuralNetwork()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(modelGRU.parameters(), lr=learning_rate)
    # Missing: return statement for modelGRU, criterion, optimizer


def train_gru_model(trainLoader, model, learning_rate=0.001, nbr_epoch=100):
    """
    Trains the GRU model with the provided training data.

    Args:
        trainLoader (DataLoader): The DataLoader containing the training dataset.
        model (GRUNeuralNetwork): The GRU model to train.
        learning_rate (float, optional): The learning rate. Defaults to 0.001.
        nbr_epoch (int, optional): The number of epochs for training. Defaults to 100.

    Returns:
        Tuple: A tuple containing the trained model, list of losses per epoch, accuracies, and epoch numbers.
    """
    modelGRU = model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(modelGRU.parameters(), lr=learning_rate)
    modelGRU.train()
    all_loss = []
    all_accuracy = []
    all_epoch = []
    for epoch in range(nbr_epoch):
        running_loss = 0.0
        total_train = 0
        correct_train = 0

        for inputs, labels in trainLoader:
            optimizer.zero_grad()
            outputs = modelGRU(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(trainLoader)
        epoch_accuracy = 100 * correct_train / total_train
        all_loss.append(epoch_loss)
        all_accuracy.append(epoch_accuracy)
        all_epoch.append(epoch + 1)

        print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')
    return modelGRU, all_loss, all_accuracy, all_epoch


def test_gru_model(testLoader, modelGRU):
    """
    Tests the GRU model on a test dataset.

    Args:
        testLoader (DataLoader): The DataLoader containing the test dataset.
        modelGRU (GRUNeuralNetwork): The GRU model to be tested.

    Returns:
        Tuple: A tuple containing the true labels, predicted labels, and test accuracy.
    """
    modelGRU.eval()
    total_test = 0
    correct_test = 0

    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in testLoader:
            labels = labels.long()
            outputs = modelGRU(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_accuracy = 100 * correct_test / total_test
    print(f'Accuracy on test set: {test_accuracy}%')
    print(classification_report(y_true, y_pred))
    return y_true, y_pred


def tune_gru_hyperparameters(model, X_valid, y_valid, output_size, hidden_size1=64, threshold=0.0001, patience=5,
                             max_epochs=50, cv=3, activation='relu',
                             verbose=1):
    """
    Tunes the hyperparameters of the GRU model using GridSearchCV.

    Args:
        model (GRUNeuralNetwork): The GRU model to be tuned.
        X_valid (Tensor): Validation dataset inputs.
        y_valid (Tensor): Validation dataset labels.
        output_size (int): Size of the output layer.
        hidden_size1 (int, optional): Size of the hidden layer in GRU. Defaults to 64.
        threshold (float, optional): Threshold for early stopping. Defaults to 0.0001.
        patience (int, optional): Patience for early stopping. Defaults to 5.
        max_epochs (int, optional): Maximum number of epochs for training. Defaults to 50.
        cv (int, optional): Number of folds in cross-validation. Defaults to 3.
        activation (str, optional): Activation function to use. Defaults to 'relu'.
        verbose (int, optional): Verbosity level. Defaults to 1.

    Returns:
        Tuple: A tuple containing the best hyperparameters and the best score.
    """
    # parameters to tune
    param_grid = {
        'module__hidden_size1': [4096, 2048],
        'optimizer__lr': [0.001],
        'module__activation': ['relu', 'sigmoid', 'tanh']
    }
    modelGRU = model
    modelGRU.eval()

    early_stopping = EarlyStopping(
        monitor='valid_acc',
        threshold=threshold,
        threshold_mode='abs',
        patience=patience,
        lower_is_better=False
    )

    classifier = NeuralNetClassifier(
        module=GRUNeuralNetwork,
        module__hidden_size1=hidden_size1,
        module__activation=activation,
        module__output_size=output_size,
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.Adam,
        max_epochs=max_epochs,
        callbacks=[early_stopping]
    )

    grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, scoring='accuracy', cv=cv, verbose=verbose, n_jobs=-1)
    grid_result = grid_search.fit(X_valid, y_valid)

    best_hyperparams = grid_search.best_params_
    best_score = grid_search.best_score_

    return best_hyperparams, best_score
