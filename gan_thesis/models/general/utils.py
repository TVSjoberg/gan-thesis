import os
import sys
import pickle
import json
import csv


def load_model(path):
    """Loads a previous model from the given path"""
    if not os.path.isfile(path):
        print('No model is saved at the specified path.')
        return
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


def save_model(model, path, force=False):
    """Save the fitted model at the given path."""
    if os.path.exists(path) and not force:
        print('The indicated path already exists. Use `force=True` to overwrite.')
        return

    base_path = os.path.dirname(path)
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    with open(path, 'wb') as f:
        pickle.dump(model, f)

    print('Model saved successfully.')


def save_json(result, path):
    """Save json at the given path."""
    base_path = os.path.dirname(path)
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    with open(path, 'w') as f:
        json.dump(result, f)


def save_csv(result, path):
    """Save csv at the given path."""
    base_path = os.path.dirname(path)
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    with open(path, 'w') as f:
        csv.dump(result, f)


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
