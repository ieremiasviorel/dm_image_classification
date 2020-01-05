import os

PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

RESULT_PLOTS_DIR = PROJECT_ROOT_DIR + '/plots'


def get_absolute_path(relative_path):
    return PROJECT_ROOT_DIR + '/' + relative_path
