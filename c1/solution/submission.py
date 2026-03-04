import argparse
import os
import time
from model import Model

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold


def get_train_data(input_dir):
    X_train = pd.read_csv(os.path.join(input_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(input_dir, 'y_train.csv'))
    return X_train, y_train


def get_test_data(input_dir):
    X_test = pd.read_csv(os.path.join(input_dir, 'X_test.csv'))
    return X_test


def preprocess(X):
    X_processed = X.copy()
    X_processed['gender'] = X_processed['gender'].astype('category')
    X_processed = pd.get_dummies(X_processed, columns=['gender'], drop_first=True, dtype=np.float64)
    X_processed['intercept'] = 1.0
    return X_processed


def get_error(model, X, y):
    y_pred = model.predict(X)
    mse = np.mean((y.values.flatten() - y_pred.flatten()) ** 2)
    return mse


def get_best_hyperparameters(X_train, y_train):
    best_mse = float('inf')
    best_params = None

    for n_components in range(2, 490, 20):
        model = Model(n_components=n_components)
        model.fit(X_train, y_train)
        mse = get_error(model, X_train, y_train)

        if mse < best_mse:
            best_mse = mse
            best_params = {'n_components': n_components}

    return best_params


def plot_n_components_vs_mse(X, y):
    n_components_list = range(2, 391, 20)
    mean_training_mses = []
    mean_testing_mses = []

    times = []

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    start_time = time.time()
    for n_components in n_components_list:
        testing_mses = []
        training_mses = []

        for train_index, val_index in kf.split(X):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            model = Model(n_components=n_components)
            model.fit(X_train, y_train)
            training_mse = get_error(model, X_train, y_train)
            testing_mse = get_error(model, X_val, y_val)
            testing_mses.append(testing_mse)
            training_mses.append(training_mse)
        end_time = time.time()
        time_taken = end_time - start_time
        times.append(time_taken)

        mean_testing_mses.append(np.mean(testing_mses))
        mean_training_mses.append(np.mean(training_mses))
        print(f'Time taken for n_components={n_components}: {time_taken:.2f} seconds.', 
              f"Expected time left: {(len(n_components_list) - len(mean_testing_mses)) * time_taken / len(mean_testing_mses):.2f} seconds")

    plt.figure(figsize=(8, 5))
    plt.plot(n_components_list, mean_training_mses, marker='o', label='Training MSE')
    plt.plot(n_components_list, mean_testing_mses, marker='s', label='Testing MSE')
    plt.title('Number of PCA Components vs MSE')
    plt.xlabel('Number of PCA Components')
    plt.ylabel('MSE')
    plt.xticks(n_components_list)
    plt.grid()
    plt.legend()
    plt.show()


def train_final_model(X_train, y_train, best_params):
    model = Model(n_components=best_params['n_components'])
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_train, y_train):
    mse = get_error(model, X_train, y_train)
    rmse = np.sqrt(mse)
    return rmse


def get_predictions(model, X_test):
    predictions = model.predict(X_test)
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    print('Reading Data')
    X_train, y_train = get_train_data(input_dir)
    X_train = preprocess(X_train)
    X_test = get_test_data(input_dir)
    X_test = preprocess(X_test)
    # plot_n_components_vs_mse(X_train, y_train)
    # print('Finding Best Hyperparameters')
    # best_params = get_best_hyperparameters(X_train, y_train)
    # print(f'Best Hyperparameters: {best_params}')
    # params = best_params
    print('Training Final Model')
    params = {'n_components': 300}
    model = train_final_model(X_train, y_train, params)
    print('Evaluating Model')
    rmse = evaluate_model(model, X_train, y_train)
    print(f'Training RMSE: {rmse}')
    print('Getting Predictions')
    prediction = get_predictions(model, X_test)

    # Make sure y_pred follows the following:
    # The file must contain exactly as many rows (excluding the header) as there are samples in X_test.
    # Each value should be a numeric age prediction (float or integer).
    # The column must be named age.

    df = pd.DataFrame(prediction, columns=['age'])

    assert len(df) == len(X_test), f"Expected {len(X_test)} predictions, but got {len(df)}."
    assert np.issubdtype(df.dtypes['age'], np.number), "Predictions must be numeric."
    assert df.columns.tolist().index('age') == 0, "The column must be named 'age' and be the first column."


    df.to_csv(os.path.join(output_dir, 'y_pred.csv'), index=False)


if __name__ == '__main__':
    main()
