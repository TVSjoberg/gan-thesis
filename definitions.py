import os


TEST_IDENTIFIER = ''

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'datasets', TEST_IDENTIFIER)
RESULT_DIR = os.path.join(ROOT_DIR, 'results', TEST_IDENTIFIER)

