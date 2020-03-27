import os
import pickle


def load(path):
    """Loads a previous model from the given path"""
    if not os.path.isfile(path):
        print('No model is saved at the specified path.')
        return
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


def save(model, path, force=False):
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
