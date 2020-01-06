import os

PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGES_DIR = PROJECT_ROOT_DIR + '/input/stanford-dogs-dataset/images/Images'
ANNOTATIONS_DIR = PROJECT_ROOT_DIR + '/input/stanford-dogs-dataset/annotations/Annotation'

RESULT_PLOTS_DIR = PROJECT_ROOT_DIR + '/plots'


def absolute_path(relative_path):
    return PROJECT_ROOT_DIR + '/' + relative_path
