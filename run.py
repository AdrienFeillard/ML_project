from data_preprocessing import *
from neural_network_model import *
from GRU_neural_network import *
from convolutional_neural_network import *
import time
from random_forest_classifier import *
from results.visualization import *
import os
import joblib


def time_elapsed(seconds):
    """
    Converts a time duration from seconds to a formatted string showing hours, minutes, and seconds.

    Args:
        seconds (float): The time duration in seconds. This can be a non-integer value to include fractional seconds.

    Returns:
        str: A formatted string representing the time duration in terms of hours, minutes, and seconds.
             The format is "{hours} hours, {minutes} minutes, {seconds} seconds", where seconds are formatted to two decimal places.
    """

    # Calculate hours by dividing the total seconds by 3600 (seconds in an hour)
    h = int(seconds // 3600)

    # Calculate remaining minutes by dividing the remainder (after subtracting hours) by 60
    m = int((seconds % 3600) // 60)

    # The remaining seconds after subtracting hours and minutes
    s = seconds % 60

    # Return the formatted time string
    return f"{h} hours, {m} minutes, {s:.2f} seconds"


def task_1_nn(save_type, model_file_path):
    """
    Performs the entire process of data preprocessing, training, tuning, and testing a neural network model for Task 1.

    Args:
        save_type (str): Indicates the type of saving operation to perform. Can be 'new' to create a new model,
                         't' to load a pre-trained model, or 'tt' to load and further train a pre-trained model.
        model_file_path (str): The file path for saving or loading the model.

    Returns:
        None: The function does not return any value but performs operations like model training, tuning, and testing.
    """
    # Model loading
    model_file = ''
    if save_type in ['t', 'tt']:
        model_file = model_file_path

    if save_type in ['t', 'tt'] and os.path.exists(model_file):
        print(f"Loading saved model from {model_file}")
        best_model = torch.load(model_file)

    start_time = time.time()

    # Data preprocessing
    print("Data Preprocessing...")
    trainLoader, testLoader, X_valid, Y_valid, output_size, input_size = data_preprocessing(model_name='NN',
                                                                                            task_name='Task 1',
                                                                                            batchsize=151,
                                                                                            temp_size=0.3,
                                                                                            test_size=0.5)
    print(f"Data preprocessing time: {time_elapsed(time.time() - start_time)}")

    if save_type == 'new':
        # Tune hyperparameters
        modelNN = NeuralNetwork(output_size)
        start_time = time.time()
        print("Tuning hyperparameters...")
        best_hyperparams, best_score = tune_nn_hyperparameters(modelNN, X_valid, Y_valid, output_size, max_epochs=50)

        print(f"Tuning time: {time_elapsed(time.time() - start_time)}")

        # Create and save the best model
        start_time = time.time()
        print("Creating and saving the best model...")
        best_model = NeuralNetwork(output_size=output_size, input_size=input_size,
                                   hidden_size1=best_hyperparams['module__hidden_size1'],
                                   hidden_size2=best_hyperparams['module__hidden_size2'])
        torch.save(best_model, './entire_modelnn_task1_tuned.pth')
        print(f"Model creation and saving time: {time_elapsed(time.time() - start_time)}")

    if save_type in ['new', 't']:
        # Train the best model and save it
        start_time = time.time()
        print("Training the best model...")
        best_model, all_loss, all_accuracy, all_epoch = train_nn_model(trainLoader, best_model, nbr_epoch=100)
        print(f"Training time: {time_elapsed(time.time() - start_time)}")

        save_training_results_to_csv(all_loss, all_accuracy, all_epoch,
                                     "./results/results_csv/train/training_result_nn_1.csv")

        start_time = time.time()
        print("Saving tuned and trained model...")
        torch.save(best_model, './entire_modelnn_task1_tuned_trained.pth')
        print(f"Model creation and saving time: {time_elapsed(time.time() - start_time)}")

    # Testing the tuned model
    start_time = time.time()
    print("Testing tuned model...")
    y_true, y_pred = test_nn_model(testLoader, best_model)
    save_test_results_to_csv(y_true, y_pred, "./results/results_csv/test/test_result_nn_1.csv")
    print(f"Testing time: {time_elapsed(time.time() - start_time)}")


def task_1_gru(save_type, model_file_path):
    """
    Performs the entire process of data preprocessing, training, tuning, and testing a GRU (Gated Recurrent Unit)
    neural network model for Task 1.

    Args:
        save_type (str): Indicates the type of saving operation to perform. Can be 'new' to create a new model,
                         't' to load a pre-trained model, or 'tt' to load and further train a pre-trained model.
        model_file_path (str): The file path for saving or loading the model.

    Returns:
        None: The function does not return any value but performs operations like model training, tuning, and testing.
    """
    # Model loading
    model_file = ''
    if save_type in ['t', 'tt']:
        model_file = model_file_path

    if save_type in ['t', 'tt'] and os.path.exists(model_file):
        print(f"Loading saved model from {model_file}")
        best_model = torch.load(model_file)

    start_time = time.time()

    # Data preprocessing
    print("Data Preprocessing...")
    trainLoader, testLoader, X_valid, Y_valid, output_size, input_size = data_preprocessing(model_name='GRU',
                                                                                            task_name='Task 1',
                                                                                            batchsize=151,
                                                                                            temp_size=0.3,
                                                                                            test_size=0.5)
    print(f"Data preprocessing time: {time_elapsed(time.time() - start_time)}")

    # Tune hyperparameters
    if save_type == 'new':
        modelGRU = GRUNeuralNetwork(input_size=input_size, output_size=output_size)

        start_time = time.time()
        print("Tuning hyperparameters...")
        best_hyperparams, best_score = tune_gru_hyperparameters(modelGRU, X_valid, Y_valid, max_epochs=50)

        print(f"Tuning time: {time_elapsed(time.time() - start_time)}")

        # Create and save the best model
        start_time = time.time()
        print("Creating and saving the best model...")
        best_model = GRUNeuralNetwork(output_size=output_size,
                                      hidden_size1=best_hyperparams['module__hidden_size1'],
                                      activation=best_hyperparams['module_activation'])
        torch.save(best_model, './entire_modelgru_task1_tuned.pth')
        print(f"Model creation and saving time: {time_elapsed(time.time() - start_time)}")

    if save_type in ['new', 't']:
        # Train the best model
        start_time = time.time()
        print("Training the best model...")
        best_model, all_loss, all_accuracy, all_epoch = train_gru_model(trainLoader, best_model, nbr_epoch=100)
        print(f"Training time: {time_elapsed(time.time() - start_time)}")
        save_training_results_to_csv(all_loss, all_accuracy, all_epoch,
                                     "./results/results_csv/train/training_result_gru_1.csv")
        start_time = time.time()
        print("Saving tuned and trained model...")
        torch.save(best_model, './entire_modelgru_task1_tuned_trained.pth')
        print(f"Model creation and saving time: {time_elapsed(time.time() - start_time)}")

    # Testing the tuned model
    start_time = time.time()
    print("Testing tuned model...")
    y_true, y_pred = test_gru_model(testLoader, best_model)
    save_test_results_to_csv(y_true, y_pred, "./results/results_csv/test/test_result_gru_1.csv")
    print(f"Testing time: {time_elapsed(time.time() - start_time)}")


def task_1_rf(save_type, model_file_path):
    """
    Performs the entire process of data preprocessing, training, tuning, and testing a Random Forest classifier model
    for Task 1.

    Args:
        save_type (str): Indicates the type of saving operation to perform. Can be 'new' to create a new model,
                         't' to load a pre-trained model, or 'tt' to load and further train a pre-trained model.
        model_file_path (str): The file path for saving or loading the model.

    Returns:
        None: The function does not return any value but performs operations like model training, tuning, and testing.
    """
    # Data preprocessing
    print("Data Preprocessing...")
    trainLoader, testLoader, X_valid, Y_valid, _, _ = data_preprocessing(model_name='RF', task_name='Task 1',
                                                                         batchsize=152, temp_size=0.3,
                                                                         test_size=0.5)
    # Model loading
    model_file = ''
    if save_type in ['t', 'tt']:
        model_file = model_file_path

    if save_type in ['t', 'tt'] and os.path.exists(model_file):
        print(f"Loading saved model from {model_file}")
        best_model = joblib.load(model_file)
        best_hyperparams = best_model.get_params()
    else:
        start_time = time.time()
        print("Tuning hyperparameters...")
        best_hyperparams, best_score = tune_rf_hyperparameters(X_valid, Y_valid)
        print(f"Tuning time: {time_elapsed(time.time() - start_time)}")

        # Create and save the best model
        start_time = time.time()
        print("Creating and saving the best model...")
        best_model = RandomForestClassifier(**best_hyperparams)
        joblib.dump(best_model, './entire_modelrf_task1_tuned.joblib')  # Saving tuned model

    # Train and Test RandomForest Model
    start_time = time.time()
    print("Training and Testing RandomForest Model...")
    y_train, y_test, Y_pred_cv, y_pred_test, accuracy_train, accuracy_test, train_tuned_model = RF_cv(trainLoader,
                                                                                                      testLoader,
                                                                                                      best_hyperparams)
    joblib.dump(train_tuned_model, './entire_modelrf_task1_trained_tuned.joblib')
    y_true = y_test
    y_pred = y_pred_test
    save_test_results_to_csv(y_true, y_pred, "./results/results_csv/test/test_result_rf_1.csv")
    print(f"Training and testing time: {time_elapsed(time.time() - start_time)}")
    print(f"Training Accuracy: {accuracy_train}")
    print(f"Testing Accuracy: {accuracy_test}")


def task_1_cnn(save_type, model_file_path):
    """
    Performs the entire process of data preprocessing, training, tuning, and testing a CNN (convolutionnal neural
    network) model for Task 1.

    Args:
        save_type (str): Indicates the type of saving operation to perform. Can be 'new' to create a new model,
                         't' to load a pre-trained model, or 'tt' to load and further train a pre-trained model.
        model_file_path (str): The file path for saving or loading the model.

    Returns:
        None: The function does not return any value but performs operations like model training, tuning, and testing.
    """
    # Model loading
    model_file = ''
    if save_type in ['t', 'tt']:
        model_file = model_file_path

    if save_type in ['t', 'tt'] and os.path.exists(model_file):
        print(f"Loading saved model from {model_file}")
        best_model = torch.load(model_file)

    start_time = time.time()

    # Data preprocessing
    print("Data Preprocessing...")
    trainLoader, testLoader, X_valid, Y_valid, output_size, input_size = data_preprocessing(model_name='CNN',
                                                                                            task_name='Task 1'
                                                                                            , batchsize=64,
                                                                                            temp_size=0.3,
                                                                                            test_size=0.5)
    print(f"Data preprocessing time: {time_elapsed(time.time() - start_time)}")
    if save_type == 'new':
        # Tune hyperparameters
        print(input_size)
        modelCNN = ConvNeuralNetwork(input_channels=input_size, kernel_size1=(3, 3, 3), kernel_size2=(3, 3, 3),
                                     output_size=output_size,
                                     hidden_size1=64,
                                     hidden_size2=128, hidden_size3=50, stride=1, padding=1)
        start_time = time.time()
        print("Tuning hyperparameters...")
        best_hyperparams, best_score = tune_cnn_hyperparameters(modelCNN, X_valid, Y_valid, output_size, max_epochs=50,
                                                                cv=3)

        print(f"Tuning time: {time_elapsed(time.time() - start_time)}")

        # Create and save the best model
        start_time = time.time()
        print("Creating and saving the best model...")
        best_model = ConvNeuralNetwork(input_channels=3,
                                       kernel_size1=best_hyperparams['module__kernel_size1'],
                                       kernel_size2=best_hyperparams['module__kernel_size2'],
                                       activation_function=best_hyperparams['module__activation_function'],
                                       output_size=output_size,
                                       hidden_size1=best_hyperparams['module__hidden_size1'],
                                       hidden_size2=best_hyperparams['module__hidden_size2'],
                                       hidden_size3=best_hyperparams['module__hidden_size3'],
                                       stride=1,
                                       padding=1)
        torch.save(best_model, './entire_modelcnn_task1_tuned.pth')
        print(f"Model creation and saving time: {time_elapsed(time.time() - start_time)}")

    if save_type in ['new', 't']:
        # Train the best model
        start_time = time.time()
        print("Training the best model...")
        best_model, all_loss, all_accuracy, all_epoch = train_cnn_model(trainLoader, best_model, nbr_epoch=100)
        print(f"Training time: {time_elapsed(time.time() - start_time)}")
        save_training_results_to_csv(all_loss, all_accuracy, all_epoch,
                                     "./results/results_csv/train/training_result_cnn_1.csv")

        start_time = time.time()
        print("Saving tuned and trained model..")
        torch.save(best_model, './entire_modelcnn_task1_tuned_trained.pth')
        print(f"Model creation and saving time: {time_elapsed(time.time() - start_time)}")

    start_time = time.time()
    print("Testing tuned model...")
    y_true, y_pred = test_cnn_model(testLoader, best_model)
    save_test_results_to_csv(y_true, y_pred, "./results/results_csv/test/test_result_cnn_1.csv")
    print(f"Testing time: {time_elapsed(time.time() - start_time)}")


def task_2_nn(save_type, model_file_path, bool_ethical=False):
    """
    Performs the entire process of data preprocessing, training, tuning, and testing a neural network model for Task 2.

    Args:
        save_type (str): Indicates the type of saving operation to perform. Can be 'new' to create a new model,
                         't' to load a pre-trained model, or 'tt' to load and further train a pre-trained model.
        model_file_path (str): The file path for saving or loading the model.

    Returns:
        None: The function does not return any value but performs operations like model training, tuning, and testing.
    """
    # Model loading
    model_file = ''
    if save_type in ['t', 'tt']:
        model_file = model_file_path

    if save_type in ['t', 'tt'] and os.path.exists(model_file):
        print(f"Loading saved model from {model_file}")
        best_model = torch.load(model_file)

    start_time = time.time()

    # Data preprocessing
    print("Data Preprocessing...")
    trainLoader, testLoader, X_valid, Y_valid, output_size, input_size = data_preprocessing(model_name='NN',
                                                                                            task_name='Task 2',
                                                                                            batchsize=151,
                                                                                            temp_size=0.3,
                                                                                            test_size=0.5)
    print(f"Data preprocessing time: {time_elapsed(time.time() - start_time)}")

    if save_type == 'new':
        # Tune hyperparameters
        modelNN = NeuralNetwork(output_size)
        start_time = time.time()
        print("Tuning hyperparameters...")
        best_hyperparams, best_score = tune_nn_hyperparameters(modelNN, X_valid, Y_valid, output_size, max_epochs=50)

        print(f"Tuning time: {time_elapsed(time.time() - start_time)}")

        # Create and save the best model
        start_time = time.time()
        print("Creating and saving the best model...")
        best_model = NeuralNetwork(output_size=output_size, input_size=input_size,
                                   hidden_size1=best_hyperparams['module__hidden_size1'],
                                   hidden_size2=best_hyperparams['module__hidden_size2'])
        torch.save(best_model, './entire_modelnn_task2_tuned.pth')
        print(f"Model creation and saving time: {time_elapsed(time.time() - start_time)}")

    if save_type in ['new', 't']:
        # Train the best model
        start_time = time.time()
        print("Training the best model...")
        best_model, all_loss, all_accuracy, all_epoch = train_nn_model(trainLoader, best_model, nbr_epoch=100)
        print(f"Training time: {time_elapsed(time.time() - start_time)}")
        save_training_results_to_csv(all_loss, all_accuracy, all_epoch,
                                     "./results/results_csv/train/training_result_nn_2.csv")

        start_time = time.time()
        print("Saving tuned and trained model..")
        torch.save(best_model, './entire_modelnn_task2_tuned_trained.pth')
        print(f"Model creation and saving time: {time_elapsed(time.time() - start_time)}")

    # Testing the tuned model
    start_time = time.time()
    print("Testing tuned model...")
    y_true, y_pred = test_nn_model(testLoader, best_model)
    if not bool_ethical:
        save_test_results_to_csv(y_true, y_pred, "./results/results_csv/test/test_result_nn_2.csv")
    else:
        save_test_results_to_csv(y_true, y_pred, "./results/results_csv/test/test_result_nn_2_ethical.csv")
    print(f"Testing time: {time_elapsed(time.time() - start_time)}")


def task_2_gru(save_type, model_file_path):
    """
    Performs the entire process of data preprocessing, training, tuning, and testing a GRU (Gated Recurrent Unit)
    neural network model for Task 2.

    Args:
        save_type (str): Indicates the type of saving operation to perform. Can be 'new' to create a new model,
                         't' to load a pre-trained model, or 'tt' to load and further train a pre-trained model.
        model_file_path (str): The file path for saving or loading the model.

    Returns:
        None: The function does not return any value but performs operations like model training, tuning, and testing.
    """
    # Model loading
    model_file = ''
    if save_type in ['t', 'tt']:
        model_file = model_file_path

    if save_type in ['t', 'tt'] and os.path.exists(model_file):
        print(f"Loading saved model from {model_file}")
        best_model = torch.load(model_file)

    start_time = time.time()

    # Data preprocessing
    print("Data Preprocessing...")
    trainLoader, testLoader, X_valid, Y_valid, output_size, input_size = data_preprocessing(model_name='GRU',
                                                                                            task_name='Task 2',
                                                                                            batchsize=151,
                                                                                            temp_size=0.3,
                                                                                            test_size=0.5)
    print(f"Data preprocessing time: {time_elapsed(time.time() - start_time)}")

    # Tune hyperparameters
    if save_type == 'new':
        modelGRU = GRUNeuralNetwork(input_size=input_size, output_size=output_size)

        start_time = time.time()
        print("Tuning hyperparameters...")
        best_hyperparams, best_score = tune_gru_hyperparameters(modelGRU, X_valid, Y_valid, max_epochs=50)

        print(f"Tuning time: {time_elapsed(time.time() - start_time)}")

        # Create and save the best model
        start_time = time.time()
        print("Creating and saving the best model...")
        best_model = GRUNeuralNetwork(output_size=output_size,
                                      hidden_size1=best_hyperparams['module__hidden_size1'],
                                      activation=best_hyperparams['module_activation'])
        torch.save(best_model, './entire_modelgru_task2_tuned.pth')
        print(f"Model creation and saving time: {time_elapsed(time.time() - start_time)}")

    if save_type in ['new', 't']:
        # Train the best model
        start_time = time.time()
        print("Training the best model...")
        best_model, all_loss, all_accuracy, all_epoch = train_gru_model(trainLoader, best_model, nbr_epoch=100)
        print(f"Training time: {time_elapsed(time.time() - start_time)}")

        save_training_results_to_csv(all_loss, all_accuracy, all_epoch,
                                     "./results/results_csv/train/training_result_gru_2.csv")

        start_time = time.time()
        print("Saving tuned and trained model...")
        torch.save(best_model, './entire_modelgru_task2_tuned_trained.pth')
        print(f"Model creation and saving time: {time_elapsed(time.time() - start_time)}")

    # Testing the tuned model
    start_time = time.time()
    print("Testing tuned model...")
    y_true, y_pred = test_gru_model(testLoader, best_model)
    save_test_results_to_csv(y_true, y_pred, "./results/results_csv/test/test_result_gru_2.csv")
    print(f"Testing time: {time_elapsed(time.time() - start_time)}")


def task_2_rf(save_type, model_file_path):
    """
    Performs the entire process of data preprocessing, training, tuning, and testing a Random Forest classifier model
    for Task 2.

    Args:
        save_type (str): Indicates the type of saving operation to perform. Can be 'new' to create a new model,
                         't' to load a pre-trained model, or 'tt' to load and further train a pre-trained model.
        model_file_path (str): The file path for saving or loading the model.

    Returns:
        None: The function does not return any value but performs operations like model training, tuning, and testing.
    """
    # Data preprocessing
    print("Data Preprocessing...")
    trainLoader, testLoader, X_valid, Y_valid, _, _ = data_preprocessing(model_name='RF', task_name='Task 2',
                                                                         batchsize=152, temp_size=0.3,
                                                                         test_size=0.5)
    # Model loading
    model_file = ''
    if save_type in ['t', 'tt']:
        model_file = model_file_path

    if save_type in ['t', 'tt'] and os.path.exists(model_file):
        print(f"Loading saved model from {model_file}")
        best_model = joblib.load(model_file)
        best_hyperparams = best_model.get_params()
    else:
        start_time = time.time()

        print("Tuning hyperparameters...")
        best_hyperparams, best_score = tune_rf_hyperparameters(X_valid, Y_valid)
        print(f"Tuning time: {time_elapsed(time.time() - start_time)}")

        # Create and save the best model
        start_time = time.time()
        print("Creating and saving the best model...")
        best_model = RandomForestClassifier(**best_hyperparams)
        joblib.dump(best_model, './entire_modelrf_task2_tuned.joblib')  # Saving tuned model

    # Train and Test RandomForest Model
    start_time = time.time()
    print("Training and Testing RandomForest Model...")
    y_train, y_test, Y_pred_cv, y_pred_test, accuracy_train, accuracy_test, train_tuned_model = RF_cv(trainLoader,
                                                                                                      testLoader,
                                                                                                      best_hyperparams)
    joblib.dump(train_tuned_model, './entire_modelrf_task2_trained_tuned.joblib')
    y_true = y_test
    y_pred = y_pred_test
    save_test_results_to_csv(y_true, y_pred, "./results/results_csv/test/test_result_rf_2.csv")

    print(f"Training and testing time: {time_elapsed(time.time() - start_time)}")
    print(f"Training Accuracy: {accuracy_train}")
    print(f"Testing Accuracy: {accuracy_test}")


def task_2_cnn(save_type, model_file_path):
    """
    Performs the entire process of data preprocessing, training, tuning, and testing a CNN (convolutionnal neural
    network) model for Task 2.

    Args:
        save_type (str): Indicates the type of saving operation to perform. Can be 'new' to create a new model,
                         't' to load a pre-trained model, or 'tt' to load and further train a pre-trained model.
        model_file_path (str): The file path for saving or loading the model.

    Returns:
        None: The function does not return any value but performs operations like model training, tuning, and testing.
    """
    # Model loading
    model_file = ''
    if save_type in ['t', 'tt']:
        model_file = model_file_path

    if save_type in ['t', 'tt'] and os.path.exists(model_file):
        print(f"Loading saved model from {model_file}")
        best_model = torch.load(model_file)

    start_time = time.time()

    # Data preprocessing
    print("Data Preprocessing...")
    trainLoader, testLoader, X_valid, Y_valid, output_size, input_size = data_preprocessing(model_name='CNN',
                                                                                            task_name='Task 2'
                                                                                            , batchsize=64,
                                                                                            temp_size=0.3,
                                                                                            test_size=0.5)
    print(f"Data preprocessing time: {time_elapsed(time.time() - start_time)}")
    if save_type == 'new':
        # Tune hyperparameters
        print(input_size)
        modelCNN = ConvNeuralNetwork(input_channels=input_size, kernel_size1=(3, 3, 3), kernel_size2=(3, 3, 3),
                                     output_size=output_size,
                                     hidden_size1=128,
                                     hidden_size2=256, hidden_size3=100, stride=1, padding=1)
        start_time = time.time()
        print("Tuning hyperparameters...")
        best_hyperparams, best_score = tune_cnn_hyperparameters(modelCNN, X_valid, Y_valid, output_size, max_epochs=50,
                                                                cv=3)

        print(f"Tuning time: {time_elapsed(time.time() - start_time)}")

        # Create and save the best model
        start_time = time.time()
        print("Creating and saving the best model...")
        best_model = ConvNeuralNetwork(input_channels=3,
                                       kernel_size1=best_hyperparams['module__kernel_size1'],
                                       kernel_size2=best_hyperparams['module__kernel_size2'],
                                       activation_function=best_hyperparams['module__activation_function'],
                                       output_size=output_size,
                                       hidden_size1=best_hyperparams['module__hidden_size1'],
                                       hidden_size2=best_hyperparams['module__hidden_size2'],
                                       hidden_size3=best_hyperparams['module__hidden_size3'],
                                       stride=1,
                                       padding=1)
        torch.save(best_model, './entire_modelcnn_task2_tuned.pth')
        print(f"Model creation and saving time: {time_elapsed(time.time() - start_time)}")

    if save_type in ['new', 't']:
        # Train the best model
        start_time = time.time()
        print("Training the best model...")
        best_model, all_loss, all_accuracy, all_epoch = train_cnn_model(trainLoader, best_model, nbr_epoch=100)
        print(f"Training time: {time_elapsed(time.time() - start_time)}")
        save_training_results_to_csv(all_loss, all_accuracy, all_epoch,
                                     "./results/results_csv/train/training_result_cnn_2.csv")
        start_time = time.time()
        print("Saving tuned and trained model..")
        torch.save(best_model, './entire_modelcnn_task2_tuned_trained.pth')
        print(f"Model creation and saving time: {time_elapsed(time.time() - start_time)}")

    start_time = time.time()
    print("Testing tuned model...")
    y_true, y_pred = test_cnn_model(testLoader, best_model)
    save_test_results_to_csv(y_true, y_pred, "./results/results_csv/test/test_result_cnn_2.csv")
    print(f"Testing time: {time_elapsed(time.time() - start_time)}")


def run_task(task, model, save_type):
    """
    Executes a specified task using a chosen model and save type.

    This function handles different tasks by selecting the appropriate model (NN, GRU, RF, CNN or Ethical)
    and managing model file paths based on the save type (tuned, tuned and trained, or new). It checks if the
    model file exists and, if not, proceeds with a new model.

    Parameters:
    :param task: An integer representing the task to be executed.
    :param model: A string representing the model type ('NN', 'GRU', 'RF', 'CNN', or 'Ethical').
    :param save_type: A string indicating the save type ('t' for tuned, 'tt' for tuned and trained, or 'new').

    Returns:
    None - The function's purpose is to execute tasks rather than return a value.
    """
    if task == 1:
        # Task 1: Handling various models for Task 1
        if model == 'NN':
            # Neural Network specific processing for Task 1
            model_file_path = ''
            if save_type == 't':
                model_file_path = './entire_modelnn_task1_tuned.pth'
            elif save_type == 'tt':
                model_file_path = './entire_modelnn_task1_tuned_trained.pth'

            if model_file_path and not os.path.exists(model_file_path):
                print(f"Model file {model_file_path} does not exist. Proceeding with a new model.")
                save_type = 'new'

            task_1_nn(save_type, model_file_path)

        elif model == 'GRU':
            # GRU specific processing for Task 1
            model_file_path = ''
            if save_type == 't':
                model_file_path = './entire_modelgru_task1_tuned.pth'
            elif save_type == 'tt':
                model_file_path = './entire_modelgru_task1_tuned_trained.pth'

            if model_file_path and not os.path.exists(model_file_path):
                print(f"Model file {model_file_path} does not exist. Proceeding with a new model.")
                save_type = 'new'

            task_1_gru(save_type, model_file_path)

        elif model == 'RF':
            # Random Forest specific processing for Task 1
            model_file_path = ''
            if save_type == 't':
                model_file_path = './entire_modelrf_task1_tuned.joblib'
            elif save_type == 'tt':
                model_file_path = './entire_modelrf_task1_tuned_trained.joblib'

            if model_file_path and not os.path.exists(model_file_path):
                print(f"Model file {model_file_path} does not exist. Proceeding with a new model.")
                save_type = 'new'

            task_1_rf(save_type, model_file_path)

        elif model == 'CNN':
            # Convolutional Neural Network specific processing for Task 1
            model_file_path = ''
            if save_type == 't':
                model_file_path = './entire_modelcnn_task1_tuned.pth'
            elif save_type == 'tt':
                model_file_path = './entire_modelcnn_task1_tuned_trained.pth'

            if model_file_path and not os.path.exists(model_file_path):
                print(f"Model file {model_file_path} does not exist. Proceeding with a new model.")
                save_type = 'new'

            task_1_cnn(save_type, model_file_path)

    elif task == 2:
        # Task 2: Handling various models for Task 2
        if model == 'NN':
            # Neural Network specific processing for Task 2
            model_file_path = ''
            if save_type == 't':
                model_file_path = './entire_modelnn_task2_tuned.pth'
            elif save_type == 'tt':
                model_file_path = './entire_modelnn_task2_tuned_trained.pth'

            if model_file_path and not os.path.exists(model_file_path):
                print(f"Model file {model_file_path} does not exist. Proceeding with a new model.")
                save_type = 'new'

            task_2_nn(save_type, model_file_path)

        elif model == 'GRU':
            # GRU specific processing for Task 2
            model_file_path = ''
            if save_type == 't':
                model_file_path = './entire_modelgru_task2_tuned.pth'
            elif save_type == 'tt':
                model_file_path = './entire_modelgru_task2_tuned_trained.pth'

            if model_file_path and not os.path.exists(model_file_path):
                print(f"Model file {model_file_path} does not exist. Proceeding with a new model.")
                save_type = 'new'

            task_2_gru(save_type, model_file_path)

        elif model == 'RF':
            # Random Forest specific processing for Task 2
            model_file_path = ''
            if save_type == 't':
                model_file_path = './entire_modelrf_task2_tuned.joblib'
            elif save_type == 'tt':
                model_file_path = './entire_modelrf_task2_tuned_trained.joblib'

            if model_file_path and not os.path.exists(model_file_path):
                print(f"Model file {model_file_path} does not exist. Proceeding with a new model.")
                save_type = 'new'

            task_2_rf(save_type, model_file_path)

        elif model == 'CNN':
            # Convolutional Neural Network specific processing for Task 2
            model_file_path = ''
            if save_type == 't':
                model_file_path = './entire_modelcnn_task2_tuned.pth'
            elif save_type == 'tt':
                model_file_path = './entire_modelcnn_task2_tuned_trained.pth'

            if model_file_path and not os.path.exists(model_file_path):
                print(f"Model file {model_file_path} does not exist. Proceeding with a new model.")
                save_type = 'new'

            task_2_cnn(save_type, model_file_path)

        elif model == 'Ethical':
            # Ethical considerations specific processing for Task 2
            model_file_path = ''
            if save_type == 't':
                model_file_path = './entire_modelnn_task2_tuned.pth'
            elif save_type == 'tt':
                model_file_path = './entire_modelnn_task2_tuned_trained.pth'

            if model_file_path and not os.path.exists(model_file_path):
                print(f"Model file {model_file_path} does not exist. Proceeding with a new model.")
                save_type = 'new'

            task_2_nn(save_type, model_file_path, True)


if __name__ == "__main__":
    while True:
        try:
            # Ask the user for the task number
            task_input = input("Enter the task number (1 or 2), 'exit' or ctrl+C to quit: ")

            # Check if the user wants to exit the program
            if task_input.lower() == 'exit':
                print("Exiting the program.")
                break

            # Validate the task number
            if task_input not in ['1', '2']:
                print("Invalid task number. Please enter a valid task number.")
                continue

            # Convert task input to an integer for further processing
            task = int(task_input)

            while True:
                # Ask the user for the model type
                model_input = input(
                    "Enter the model (NN, GRU, RF, CNN, Ethical), 'exit' to come back to the task choice: ")

                # Check if the user wants to exit the program
                if model_input.lower() == 'exit':
                    print("Choose again the task.")
                    break  # Break out of the while loop to exit the program

                # Check for valid model
                if model_input not in ['NN', 'GRU', 'RF', 'CNN', 'Ethical']:
                    print("Invalid model type. Please enter a valid model.")
                    continue  # Continue to the next iteration of the loop

                while True:
                    save_type_input = input("Do you want to use a saved model? Enter 'yes' or 'no'. If you have "
                                            "already existing model files 'no' will create new ones and destroy the "
                                            "existing ones (If you want to keep the already existing ones change the "
                                            "names of the existing files): ")

                    if save_type_input.lower() == 'yes':
                        save_type_choice = input("Choose the save type of the model you want to use: 't' for model "
                                                 "tuned with the best parameters or 'tt' for model tuned  and trained "
                                                 "with best parameters: ")

                        if save_type_choice.lower() not in ['t', 'tt']:
                            print("Invalid saved model choice. Please enter a valid choice.")
                            continue
                        elif save_type_choice == 't':
                            save_type = 't'
                        else:
                            save_type = 'tt'

                    elif save_type_input.lower() == 'no':
                        save_type = 'new'
                    else:
                        print("Invalid response type 'yes' or 'no'")
                        continue
                    if save_type not in ['t', 'tt', 'no', 'new']:
                        print('sd')
                    else:
                        run_task(task, model_input, save_type)
                    break
            break


        except ValueError:
            print("An error occurred.")
        except Exception as e:
            # Handle other potential exceptions
            print(f"An error occurred: {e}")
        except KeyboardInterrupt:
            print("\nUser interrupted the program. Exiting...")
            break
    # List to hold the names of the relevant CSV files
    csv_training_files = []
    csv_test_files = []
    # Directory where the files are located (assuming current directory)
    train_directory = './results/results_csv/train'
    test_directory = './results/results_csv/test'
    # Loop through all files in the directory
    for file in os.listdir(train_directory):
        if file.startswith('training_result_') and file.endswith('.csv'):
            csv_training_files.append(os.path.join(train_directory, file))

    # Check if we found any files
    if csv_training_files:
        # Make sure you have imported plot_accuracy_for_networks from visualization.py
        plot_accuracy_for_networks(csv_training_files)
    else:
        print("No training result CSV files found.")

    for file in os.listdir(test_directory):
        if file.startswith('test_result_') and file.endswith('.csv'):
            csv_test_files.append(os.path.join(test_directory, file))

    # Check if we found any files
    if csv_test_files:
        # Make sure you have imported plot_accuracy_for_networks from visualization.py
        plot_and_save_confusion_matrix(csv_test_files)
    else:
        print("No test result CSV files found.")

    csv_test_file = os.path.join(test_directory, 'test_result_nn_2_ethical.csv')

    # Check if the specific test file exists
    if os.path.exists(csv_test_file):
        # Call the plotting function for just that file
        plot_and_save_confusion_matrix([csv_test_file], title="Confusion Matrix for NN Task 2 Ethical",
                                       save_path="./results/confusion_matrix_nn_2_ethical.png")
    else:
        print("The file test_result_nn_2_ethical.csv does not exist in the test directory.")
# %%
