import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from skorch.callbacks import EarlyStopping
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


class ConvNeuralNetwork(nn.Module):
    """
    A Convolutional Neural Network using 3D convolutions for processing 3D data.

    Attributes:
        conv1 (nn.Conv3d): First 3D convolution layer.
        pool (nn.MaxPool3d): Max pooling layer.
        conv2 (nn.Conv3d): Second 3D convolution layer.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        activation_function (Callable): Activation function.

    Args:
        input_channels (int, optional): Number of input channels. Defaults to 3.
        kernel_size1 (tuple, optional): Kernel size for the first convolution layer. Defaults to (3, 3, 3).
        kernel_size2 (tuple, optional): Kernel size for the second convolution layer. Defaults to (3, 3, 3).
        activation_function (Callable, optional): Activation function to use. Defaults to F.relu.
        output_size (int, optional): Size of the output layer. Defaults to 7.
        hidden_size1 (int, optional): Size of the first hidden layer. Defaults to 64.
        hidden_size2 (int, optional): Size of the second hidden layer. Defaults to 128.
        hidden_size3 (int, optional): Size of the third hidden layer. Defaults to 50.
        stride (int, optional): Stride for convolution layers. Defaults to 1.
        padding (int, optional): Padding for convolution layers. Defaults to 1.
    """

    def __init__(self, input_channels=3, kernel_size1=(3, 3, 3), kernel_size2=(3, 3, 3), activation_function=F.relu, output_size=7, hidden_size1=64, hidden_size2=128, hidden_size3=50, stride=1, padding=1):
        super(ConvNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=input_channels, out_channels=hidden_size1, kernel_size=kernel_size1, stride=stride, padding=padding)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv3d(in_channels=hidden_size1, out_channels=hidden_size2, kernel_size=kernel_size2, stride=stride, padding=padding)
        self._to_linear = None
        self._calculate_to_linear((input_channels, 3, 3, 11))  # Example input shape
        self.fc1 = nn.Linear(in_features=self._to_linear, out_features=hidden_size3)
        self.fc2 = nn.Linear(in_features=hidden_size3, out_features=output_size)
        self.activation_function = activation_function

    def _calculate_to_linear(self, input_shape):
        """
        Helper method to calculate the number of features before the fully connected layer.

        Args:
            input_shape (tuple): The shape of the input.
        """
        with torch.no_grad():
            input_tensor = torch.zeros((1, *input_shape))
            output_tensor = self.conv1(input_tensor)
            output_tensor = self.pool(output_tensor)
            output_tensor = self.conv2(output_tensor)
            self._to_linear = int(torch.prod(torch.tensor(output_tensor.shape[1:])))

    def forward(self, x):
        """
        Forward pass of the neural network.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: The output of the network.
        """
        x = self.activation_function(self.conv1(x))
        x = self.pool(x)
        x = self.activation_function(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.activation_function(self.fc1(x))
        x = self.fc2(x)
        return x


def restart_cnn_model(learning_rate=0.0001):
    """
    Initializes the Convolutional Neural Network with the specified learning rate.

    Args:
        learning_rate (float): The learning rate for the optimizer.

    Returns:
        tuple: The initialized CNN model, loss criterion, and optimizer.
    """
    modelCNN = ConvNeuralNetwork()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(modelCNN.parameters(), lr=learning_rate)
    return modelCNN, criterion, optimizer


def train_cnn_model(trainLoader, model, learning_rate=0.0001, nbr_epoch=100):
    """
    Trains the CNN model with the provided training data.

    Args:
        trainLoader (DataLoader): DataLoader containing the training dataset.
        model (ConvNeuralNetwork): The CNN model to train.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.0001.
        nbr_epoch (int, optional): Number of training epochs. Defaults to 100.

    Returns:
        Tuple: Trained model, list of losses per epoch, list of accuracies, and list of epochs.
    """
    modelCNN = model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(modelCNN.parameters(), lr=learning_rate)
    modelCNN.train()

    all_loss = []
    all_accuracy = []
    all_epoch = []

    for epoch in range(nbr_epoch):
        running_loss = 0.0
        total_train = 0
        correct_train = 0

        for inputs, labels in trainLoader:
            optimizer.zero_grad()
            outputs = modelCNN(inputs)
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
    return modelCNN, all_loss, all_accuracy, all_epoch


def test_cnn_model(testLoader, model):
    """
    Tests the CNN model on a test dataset.

    Args:
        testLoader (DataLoader): DataLoader containing the test dataset.
        model (ConvNeuralNetwork): The CNN model to be tested.

    Returns:
        Tuple: A tuple containing the true labels, predicted labels, and test accuracy.
    """
    modelCNN = model
    modelCNN.eval()
    total_test = 0
    correct_test = 0

    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in testLoader:
            labels = labels.long()
            outputs = modelCNN(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_accuracy = 100 * correct_test / total_test
    print(f'Accuracy on test set: {test_accuracy}%')
    print(classification_report(y_true, y_pred))
    return y_true, y_pred


def tune_cnn_hyperparameters(model, X_valid, y_valid, output_size, kernel_size1=(3, 3, 3), kernel_size2=(3, 3, 3), activation_function=F.relu,
                             hidden_size1=64, hidden_size2=128, hidden_size3=50, nbr_features=99,
                             threshold=0.001, patience=5, max_epochs=2, cv=3, verbose=2):
    """
    Tunes the hyperparameters of the CNN model using GridSearchCV.

    Args:
        model (ConvNeuralNetwork): The CNN model to be tuned.
        X_valid (Tensor): Validation dataset inputs.
        y_valid (Tensor): Validation dataset labels.
        output_size (int): Size of the output layer.
        kernel_size1 (tuple, optional): Kernel size for the first convolution layer. Defaults to (3, 3, 3).
        kernel_size2 (tuple, optional): Kernel size for the second convolution layer. Defaults to (3, 3, 3).
        activation_function (Callable, optional): Activation function to use. Defaults to F.relu.
        hidden_size1 (int, optional): Size of the first hidden layer. Defaults to 64.
        hidden_size2 (int, optional): Size of the second hidden layer. Defaults to 128.
        hidden_size3 (int, optional): Size of the third hidden layer. Defaults to 50.
        nbr_features (int, optional): Number of features. Defaults to 99.
        threshold (float, optional): Threshold for early stopping. Defaults to 0.001.
        patience (int, optional): Patience for early stopping. Defaults to 5.
        max_epochs (int, optional): Maximum number of epochs for training. Defaults to 2.
        cv (int, optional): Number of folds in cross-validation. Defaults to 3.
        verbose (int, optional): Verbosity level. Defaults to 2.

    Returns:
        Tuple: A tuple containing the best hyperparameters and the best score.
    """
    # parameters to tune
    param_grid = {
        'module__hidden_size1': [64],
        'module__hidden_size2': [128],
        'module__hidden_size3': [50],
        'module__kernel_size': [(3, 3, 3),(2, 2, 2)],
        "module__activation_function": [F.relu,F.elu],
        'optimizer__lr': [0.0001]
    }
    param_grid2 = {
        'module__hidden_size1': [64],
        'module__hidden_size2': [128],
        'module__hidden_size3': [50],
        'module__kernel_size1': [(1, 1, 1),(2, 2, 2)],
        'module__kernel_size2': [(3, 3, 3),(2, 2, 2)],
        "module__activation_function": [F.relu,F.leaky_relu],
        'optimizer__lr': [0.0001]
    }
    param_grid3 = {
        'module__hidden_size1': [64],
        'module__hidden_size2': [128],
        'module__hidden_size3': [50],
        'module__kernel_size1': [(2, 2, 2)],
        'module__kernel_size2': [(2, 2, 2)],
        "module__activation_function": [F.relu],
        'optimizer__lr': [0.0001]
    }

    modelCNN = model
    modelCNN.eval()

    early_stopping = EarlyStopping(
        monitor='valid_acc',  # Change to 'valid_acc' for accuracy
        threshold=0.001,  # Define your threshold
        threshold_mode='abs',  # 'rel' for relative, 'abs' for absolute
        patience=3,
        lower_is_better=False# Number of epochs to wait after condition is met
    )

    # Convert the PyTorch model to a skorch classifier to use in GridSearchCV
    classifier = NeuralNetClassifier(
        module=ConvNeuralNetwork,
        module__hidden_size1=hidden_size1,
        module__hidden_size2=hidden_size2,
        module__hidden_size3=hidden_size3,
        module__kernel_size1=kernel_size1,
        module__kernel_size2=kernel_size2,
        module__output_size=output_size,
        module__activation_function=activation_function,
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.Adam,
        max_epochs=max_epochs,  # or choose an appropriate number of epochs
        callbacks=[early_stopping]
    )

    # Use GridSearchCV for hyperparameter tuning, cv for the number of folds in cross-validation, verbose for the explicit stage of tuning
    grid_search = GridSearchCV(classifier, param_grid2, scoring='accuracy', cv=cv, verbose=verbose)
    # get grid result
    grid_result = grid_search.fit(X_valid, y_valid)

    # Get the best hyperparameters
    best_hyperparams = grid_search.best_params_
    print(best_hyperparams)
    # get best score
    best_score = grid_search.best_score_

    return best_hyperparams, best_score
