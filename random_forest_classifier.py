import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


def RF_cv(trainLoader, testLoader, best_params):
    """
    Trains a RandomForestClassifier with given best parameters on data from a PyTorch DataLoader and evaluates its performance.

    Args:
        trainLoader (DataLoader): DataLoader for training data.
        testLoader (DataLoader): DataLoader for testing data.
        best_params (dict): Best hyperparameters for the RandomForestClassifier.

    Returns:
        tuple: A tuple containing training labels, test labels, cross-validation predictions, 
               test predictions, training accuracy, testing accuracy, and the trained classifier.
    """
    # Combining the data from the trainLoader
    X_train, Y_train = [], []
    for data in trainLoader:
        inputs, labels = data
        X_train.append(inputs)
        Y_train.append(labels)

    X_train = torch.cat(X_train, 0).numpy()
    Y_train = torch.cat(Y_train, 0).numpy()

    # Training the RandomForest Classifier with best parameters
    rf_classifier = RandomForestClassifier(**best_params, verbose=2)
    Y_pred_cv = cross_val_predict(rf_classifier, X_train, Y_train, cv=5)
    rf_classifier.fit(X_train, Y_train)

    # Combining the data from the testLoader
    X_test = []
    Y_test = []
    for data in testLoader:
        inputs, labels = data
        X_test.append(inputs)
        Y_test.append(labels)

    X_test = torch.cat(X_test, 0).numpy()
    Y_test = torch.cat(Y_test, 0).numpy()

    # Predicting on the test set
    y_pred_test = rf_classifier.predict(X_test)
    accuracy_train = accuracy_score(Y_train, Y_pred_cv)
    accuracy_test = accuracy_score(Y_test, y_pred_test)
    print(classification_report(Y_test, y_pred_test))
    return Y_train, Y_test, Y_pred_cv, y_pred_test, accuracy_train, accuracy_test, rf_classifier


def tune_rf_hyperparameters(X_validation, Y_validation, cv=4, verbose=2, n_jobs=-1):
    """
    Tunes the hyperparameters of a RandomForestClassifier using GridSearchCV.

    Args:
        X_validation (array-like): Validation dataset inputs.
        Y_validation (array-like): Validation dataset labels.
        cv (int, optional): Number of folds in cross-validation. Defaults to 4.
        verbose (int, optional): Verbosity level for GridSearchCV. Defaults to 2.
        n_jobs (int, optional): Number of jobs to run in parallel for GridSearchCV. Defaults to -1 (using all processors).

    Returns:
        tuple: A tuple containing the best hyperparameters and the best score from GridSearchCV.
    """
    # Define the parameter grid to test
    param_grid = {
        'n_estimators': [50, 100, 200],
        'random_state': [100],
        'max_depth': [None, 10],
    }
    param_grid2 = {
        'n_estimators': [50],
        'random_state': [100],
        'max_depth': [None, 10],
    }
    # Initialize RandomForest model
    rf = RandomForestClassifier()

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv, verbose=verbose, n_jobs=n_jobs)

    # Execute the grid search
    grid_search.fit(X_validation, Y_validation)

    return grid_search.best_params_, grid_search.best_score_
