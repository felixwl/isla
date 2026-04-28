import argparse
import os
from model import Model

import numpy as np
import pandas as pd

SUBJECTS = ['A', 'B', 'C', 'D', 'E', 'F']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    for subject in SUBJECTS:
        print(f'Subject {subject}: loading data')
        X_train = np.load(os.path.join(input_dir, f'subject_{subject}_X_train.npy'))
        y_train = np.load(os.path.join(input_dir, f'subject_{subject}_y_train.npy'))
        X_test = np.load(os.path.join(input_dir, f'subject_{subject}_X_test.npy'))

        m = Model()
        print(f'Subject {subject}: training')
        m.fit(X_train, y_train)
        print(f'Subject {subject}: predicting')
        y_pred = m.predict(X_test)

        out_path = os.path.join(output_dir, f'subject_{subject}_y_pred.csv')
        pd.DataFrame({'y_pred': y_pred}).to_csv(out_path, index=False)
        print(f'Subject {subject}: done ({len(y_pred)} predictions saved)')

    # zip the output directory
    import shutil
    shutil.make_archive(output_dir, 'zip', output_dir)
    print(f'Output directory "{output_dir}" zipped to "{output_dir}.zip"')


if __name__ == '__main__':
    main()
