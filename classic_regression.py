import os
import sys
import time
import joblib
import logging
import argparse
import numpy as np
from utils import *
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from multiprocessing import Process, Queue
from sklearn.metrics import mean_squared_error, r2_score

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

pca = None


def parse_option():
    parser = argparse.ArgumentParser('Classical regression approach for concentration estimation', add_help=False)
    parser.add_argument('--train-path', required=True, type=str, default="numpy_features/TRAIN_features.npy",
                        help='Path to the train .npy file')
    parser.add_argument('--test-path', required=True, type=str, default="numpy_features/TEST_features.npy",
                        help='Path to the test .npy file')
    parser.add_argument('--logfile-name', type=str, required=True, help='path to save the log file')
    parser.add_argument('--use-pca', type=bool, default=False, help='Use PCA to reduce dimensionality')

    args, unparsed = parser.parse_known_args()
    return args


def get_models():
    models = []

    from sklearn import tree
    model_DecisionTreeRegressor = tree.DecisionTreeRegressor()

    from sklearn import linear_model
    model_LinearRegression = linear_model.LinearRegression()

    from sklearn import neighbors
    model_KNeighborsRegressor = neighbors.KNeighborsRegressor()

    from sklearn import ensemble
    model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=100)

    model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators=100)

    from sklearn.ensemble import BaggingRegressor
    model_BaggingRegressor = BaggingRegressor()

    from sklearn.tree import ExtraTreeRegressor
    model_ExtraTreeRegressor = ExtraTreeRegressor()

    models.append(model_AdaBoostRegressor)
    models.append(model_BaggingRegressor)
    models.append(model_DecisionTreeRegressor)
    models.append(model_ExtraTreeRegressor)
    models.append(model_KNeighborsRegressor)
    models.append(model_LinearRegression)
    models.append(model_RandomForestRegressor)

    return models


def load_numpy_features(numpy_path, is_train, use_pca, n_components):
    global pca

    data_dict = np.load(numpy_path, allow_pickle=True).item()

    images = data_dict['images']
    labels = data_dict['labels']

    # Reshape 2D images into 1D vector
    images_features = images.reshape(images.shape[0], -1)

    # Apply PCA if flag enabled
    if use_pca:

        # Create a new model if it doesn't exist
        if pca is None:
            pca = PCA(n_components=n_components)

            # Fit the model only with training data
            if is_train:
                pca.fit(images_features)

        # Transform data to PCA space
        images_features = pca.transform(images_features)

    return images_features, labels


def load_and_split_numpy_features(data_paths, test_labels, use_pca, n_components):
    global pca
    pca = None

    all_features = []
    all_labels = []

    for data_path in data_paths:
        features, labels = load_numpy_features(data_path, is_train=True, use_pca=use_pca, n_components=n_components)
        print(features.shape)
        all_features.append(features)
        all_labels.append(labels)

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Split data into train and test sets
    train_idx = np.isin(all_labels, test_labels, invert=True)
    test_idx = np.isin(all_labels, test_labels)

    X_train = all_features[train_idx]
    y_train = all_labels[train_idx]
    X_test = all_features[test_idx]
    y_test = all_labels[test_idx]

    return X_train, X_test, y_train, y_test


def save_plot(y_test, combined_predictions):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, combined_predictions, color='red', alpha=0.7, label='Predicted Label')
    plt.scatter(y_test, y_test, color='blue', alpha=0.7, label='True Label')
    plt.legend()
    plt.xlabel('True label')
    plt.ylabel('Predicted label')
    plt.title('True vs Predicted Values')
    plt.savefig('/TEST.png')


def run_model(model, model_name, X_train, y_train, X_test, y_test, result_queue):
    dirs = 'logs'
    os.makedirs(dirs, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(dirs, f"{args.logfile_name}_{current_time}.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    logging.captureWarnings(True)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.WARNING)  # Set handler level to capture WARNING and above
    logger.addHandler(handler)

    # Add another StreamHandler for console output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # Set console handler level to capture INFO and above
    logger.addHandler(console_handler)

    # Logging the arguments
    logger.info('Arguments:')
    for arg in vars(args):
        logger.info(f'{arg}: {getattr(args, arg)}')

    logger.info(f'\nMODEL Name: {model_name}')
    logging.info('Log DATE: %s', datetime.now())

    # run the model on the train and test data
    time_start = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    time_end = time.time()

    result_queue.put((model_name, y_pred))

    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    logger.info(f'\nDataFrame Contents for model {model_name}:\n%s', df.to_string(index=False))

    logger.info(f'Time for training the model {model_name}: {time_end - time_start}')

    print(f'Finished for model {model_name}.')


if __name__ == "__main__":
    args = parse_option()

    # Load training and test data
    X_train, y_train = load_numpy_features(args.train_path, is_train=True, use_pca=args.use_pca,
                                           n_components=100)
    X_test, y_test = load_numpy_features(args.test_path, is_train=False, use_pca=args.use_pca,
                                         n_components=100)

    models = get_models()

    model_names = ["AdaBoost Regressor", "Bagging Regressor", "Decision Tree Regressor", "Extra Tree Regressor",
                   "KNeighbors Regressor", "Linear Regression", "Random ForestRegressor"]

    all_predictions = [[] for _ in range(len(models))]  # List of lists
    processes = []

    start_time = time.time()  # Record the start time
    result_queue = Queue()

    for num, (model, model_name) in enumerate(zip(models, model_names)):
        p = Process(target=run_model, args=(model, model_name, X_train, y_train, X_test, y_test, result_queue))
        processes.append(p)
        p.start()

    # Collect predictions from the Queue
    all_predictions = {}
    for _ in range(len(models)):
        model_name, y_pred = result_queue.get()  # Get the model name and predictions from the Queue
        all_predictions[model_name] = y_pred

    # Combine predictions from all models
    combined_predictions = np.vstack(list(all_predictions.values())).mean(axis=0)

    # Calculate RMSE and R2 scores using combined predictions
    rmse = np.sqrt(mean_squared_error(y_test, combined_predictions))
    r2 = r2_score(y_test, combined_predictions)

    save_plot(y_test, combined_predictions)

    df = pd.DataFrame({'Actual': y_test, 'Predicted': combined_predictions})
    print(f'\nDataFrame Contents for model {model_name}:\n%s', df.to_string(index=False))

    print("Average RMSE:", rmse)
    print("Average R2 Score:", r2)