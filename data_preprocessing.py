import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA


def data_preprocessing(model_name, task_name, batchsize, temp_size=0.3, test_size=0.5, num_channels=3):
    """
    Process data for machine learning models by encoding labels, reshaping, and splitting the dataset.

    Parameters:
    - model_name (str): Name of the model (e.g., 'NN', 'CNN', 'RF', 'GRU', 'Ethical').
    - task_name (str): Specific task to be performed ('Task 1' or 'Task 2').
    - batchsize (int): Batch size for data loading.
    - temp_size (float, optional): Fraction of the dataset to be used as temporary set. Defaults to 0.3.
    - test_size (float, optional): Fraction of the temporary set to be used as test set. Defaults to 0.5.
    - num_channels (int, optional): Number of channels for CNN. Defaults to 3.

    Returns:
    - trainLoader (DataLoader): DataLoader object for training data.
    - testLoader (DataLoader): DataLoader object for testing data.
    - X_validation (Tensor): Validation input data.
    - Y_validation (Tensor): Validation target data.
    - output_size (int): Number of unique labels.
    - input_size (int): Input feature size.
    """
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Load and clean data
    df = pd.read_parquet("./dataset/All_Relative_Results_Cleaned.parquet")
    df_clean = df.dropna()
    index = df_clean.columns.get_loc('time(s)')

    # Encode labels based on task
    if task_name == 'Task 1':
        Y = df_clean['Exercise']
    elif task_name == 'Task 2':
        Y = df_clean['Exercise'] + 'with mistakes' + df_clean['Set']
    else:
        raise ValueError("Value Error: type 'Task 1' or 'Task 2' ")

    # Encode the labels into a numeric format

    label_encoder = LabelEncoder()
    Y_encoded = label_encoder.fit_transform(Y)
    output_size = len(np.unique(Y_encoded))

    # Prepare data based on model type
    shuffle = False
    if model_name in ['NN', 'CNN', 'RF', 'Ethical']:
        # We ignore the 'time(s)' column for these models
        index += 1
        shuffle = True  # Shuffle data during training
    elif model_name == 'GRU':
        # Include 'time(s)', 'Participant' and 'Camera' as features for GRU
        add_participants = df_clean['Participant']
        add_camera = df_clean['Camera']
        add_participants_encoded = label_encoder.fit_transform(add_participants)
        add_camera_encoded = label_encoder.fit_transform(add_camera)

    # Extract the feature columns for training
    X = df_clean.iloc[:, index:]

    # Data preparation for different model types
    if model_name in ['NN', 'RF', 'Ethical']:
        # For non-sequential models, we use a simple float tensor
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        input_size = X.shape[1]
    elif model_name == 'GRU':
        X = X.assign(add_participants_encoded=add_participants_encoded, add_camera_encoded=add_camera_encoded)
        input_size = X.shape[1]
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
    elif model_name == 'CNN':
        # Specific processing for CNN model
        # Extract columns for x, y, z coordinates
        cols_x = [col for col in X.columns if col.endswith('x')]
        cols_y = [col for col in X.columns if col.endswith('y')]
        cols_z = [col for col in X.columns if col.endswith('z')]

        nbr_of_points = (X.shape[1]) // 3
        input_size = nbr_of_points

        # Calculate the mean coordinates and perform PCA for dimensionality reduction
        data_x = X[cols_x].values
        data_y = X[cols_y].values
        data_z = X[cols_z].values

        mean_x = np.mean(data_x, axis=0)
        mean_y = np.mean(data_y, axis=0)
        mean_z = np.mean(data_z, axis=0)

        mean_coords = np.stack([mean_x, mean_y, mean_z], axis=1)
        pca = PCA(n_components=3)
        reduced_coords = pca.fit_transform(mean_coords)

        # Normalize and map the reduced coordinates to a 3D grid
        normalized_coords = (reduced_coords - reduced_coords.min(0)) / reduced_coords.ptp(0)
        grid_coords = np.round(normalized_coords * np.array([2, 2, 10])).astype(int)
        depth, height, width = 3, 3, 11

        num_samples = X.shape[0]
        grid_tensor = torch.zeros((num_samples, num_channels, depth, height, width))
        # Populate the grid tensor with the original features
        for feature_idx in range(33):  # There is 33 features for CNN for the 33 different joints
            d, h, w = grid_coords[feature_idx]

            d, h, w = np.clip([d, h, w], 0, [depth - 1, height - 1, width - 1])

            grid_tensor[:, 0, d, h, w] = torch.tensor(X[cols_x[feature_idx]].values)
            grid_tensor[:, 1, d, h, w] = torch.tensor(X[cols_y[feature_idx]].values)
            grid_tensor[:, 2, d, h, w] = torch.tensor(X[cols_z[feature_idx]].values)

            X_tensor = grid_tensor

    # Convert the encoded labels into a tensor
    Y_tensor = torch.tensor(Y_encoded, dtype=torch.long)

    # Split data into training, validation, and testing sets
    X_train, X_temp, Y_train, Y_temp = train_test_split(X_tensor, Y_tensor, test_size=temp_size)
    X_validation, X_test, Y_validation, Y_test = train_test_split(X_temp, Y_temp, test_size=test_size)
    if model_name == 'Ethical':
        X_test = 0.87 * X_test
    # Create DataLoader objects
    train_dataset = TensorDataset(X_train, Y_train)
    # validation_dataset = TensorDataset(X_validation,Y_validation)
    test_dataset = TensorDataset(X_test, Y_test)
    trainLoader = DataLoader(train_dataset, batch_size=batchsize, shuffle=shuffle)
    # validationLoader = DataLoader(validation_dataset,batch_size=batchsize,shuffle=shuffle)
    testLoader = DataLoader(test_dataset, batch_size=batchsize, shuffle=shuffle)

    # Return the DataLoader objects along with validation data, output size, and input size
    return trainLoader, testLoader, X_validation, Y_validation, output_size, input_size
