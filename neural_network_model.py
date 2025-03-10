import torch
import torch.nn as nn
import torch.optim as optim
from skorch.callbacks import EarlyStopping
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

class NeuralNetwork(nn.Module):
    """
    A simple feedforward neural network for classification tasks.

    Attributes:
        layer1 (nn.Linear): The first linear layer.
        activation (Callable): Activation function.
        layer2 (nn.Linear): The second linear layer.
        layer3 (nn.Linear): The third linear layer.
        outputsize (int): Size of the output layer.

    Args:
        output_size (int): The size of the output layer.
        input_size (int, optional): The size of the input layer. Defaults to 99.
        hidden_size1 (int, optional): The size of the first hidden layer. Defaults to 64.
        hidden_size2 (int, optional): The size of the second hidden layer. Defaults to 64.
        activation (str, optional): Type of activation function to use. Defaults to 'relu'.
    """
    def __init__(self, output_size, input_size=99, hidden_size1=64, hidden_size2=64, activation='relu'):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.activation = self.get_activation(activation)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.layer3 = nn.Linear(hidden_size2, output_size)
        self.reset_parameters()
        self.outputsize = output_size

    def forward(self, x):
        """
        Forward pass of the neural network.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: The output of the network.
        """
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.layer3(x)
        return x

    def reset_parameters(self):
        """
        Resets the parameters of the network layers.
        """
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

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
        elif activation == 'softmax':
            return nn.Softmax()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")


def restart_nn_model(output_size, learning_rate=0.001):
    """
    Initializes and restarts the neural network model with a given output size and learning rate.

    Args:
        output_size (int): The size of the output layer of the neural network.
        learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.001.

    Returns:
        tuple: A tuple containing the initialized model, criterion, and optimizer.
    """

    model = NeuralNetwork(output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def train_nn_model(trainLoader, model, learning_rate=0.001, nbr_epoch=100):
    """
    Trains the neural network model using the provided training data.

    Args:
        trainLoader (DataLoader): DataLoader containing the training dataset.
        model (NeuralNetwork): The neural network model to be trained.
        learning_rate (float, optional): The learning rate. Defaults to 0.001.
        nbr_epoch (int, optional): The number of epochs for training. Defaults to 100.

    Returns:
        tuple: A tuple containing the trained model, list of losses per epoch, accuracies, and epoch numbers.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    all_loss = []
    all_accuracy = []
    all_epoch = []

    for epoch in range(nbr_epoch):
        running_loss = 0.0
        total_train = 0
        correct_train = 0

        for inputs, labels in trainLoader:
            optimizer.zero_grad()
            outputs = model(inputs)
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
    return model, all_loss, all_accuracy, all_epoch


def test_nn_model(testLoader, model):
    """
    Tests the neural network model using the provided test data.

    Args:
        testLoader (DataLoader): DataLoader containing the test dataset.
        model (NeuralNetwork): The neural network model to be tested.

    Returns:
        tuple: A tuple containing the true labels, predicted labels, test accuracy, and a classification report.
    """
    model.eval()
    total_test = 0
    correct_test = 0

    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in testLoader:
            labels = labels.long()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_accuracy = 100 * correct_test / total_test
    print(f'Accuracy on test set: {test_accuracy}%')
    print(classification_report(y_true, y_pred))
    return y_true, y_pred


def tune_nn_hyperparameters(model, X_valid, y_valid, output_size, hidden_size1=64, hidden_size2=64, threshold=0.0001,
                            patience=5, max_epochs=50, cv=3, activation ='relu',
                            verbose=1):
    """
    Tunes the hyperparameters of the neural network using GridSearchCV.

    Args:
        model (NeuralNetwork): The neural network model.
        X_valid (array): Validation dataset inputs.
        y_valid (array): Validation dataset labels.
        output_size (int): Size of the output layer.
        hidden_size1 (int, optional): Size of the first hidden layer. Defaults to 64.
        hidden_size2 (int, optional): Size of the second hidden layer. Defaults to 64.
        threshold (float, optional): Threshold for early stopping. Defaults to 0.0001.
        patience (int, optional): Patience for early stopping. Defaults to 5.
        max_epochs (int, optional): Maximum number of epochs for training. Defaults to 50.
        cv (int, optional): Number of folds in cross-validation. Defaults to 3.
        activation (str, optional): Activation function to use. Defaults to 'relu'.
        verbose (int, optional): Verbosity level. Defaults to 1.

    Returns:
        tuple: A tuple containing the best hyperparameters and the best score.
    """
    # parameters to tune
    """
    param_grid = {
        'module__hidden_size1': [4096, 2048],
        'module__hidden_size2': [512],
        'batch_size': [153, 150, 152, 151],
        'optimizer__lr': [0.001]
    }
    """
    param_grid = {
        'module__hidden_size1': [4096, 2048],
        'module__hidden_size2': [512, 384],
        'optimizer__lr': [0.001],
        'module__activation':['relu','tanh', 'sigmoid']

    }
    model.eval()
    print(model.outputsize)
    early_stopping = EarlyStopping(
        monitor='valid_loss',  # Change to 'valid_acc' for accuracy
        threshold=threshold,  # Define your threshold
        threshold_mode='rel',  # 'rel' for relative, 'abs' for absolute
        patience=patience  # Number of epochs to wait after condition is met
    )

    # Convert the PyTorch model to a skorch classifier to use in GridSearchCV
    classifier = NeuralNetClassifier(
        module=NeuralNetwork,
        module__hidden_size1=hidden_size1,  # Example values
        module__hidden_size2=hidden_size2,
        module__output_size=output_size,
        module__activation=activation,
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.Adam,
        max_epochs=max_epochs,  # or choose an appropriate number of epochs
        callbacks=[early_stopping]

    )

    # Use GridSearchCV for hyperparameter tuning, cv for the number of folds in cross-validation, verbose for the explicit stage of tuning
    grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, scoring='accuracy', cv=cv, verbose=verbose)
    # get grid result
    grid_search.fit(X_valid, y_valid)

    # Get the best hyperparameters
    best_hyperparams = grid_search.best_params_

    # get best score
    best_score = grid_search.best_score_

    return best_hyperparams, best_score
