import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix, classification_report

from definitions import absolute_path, RESULT_PLOTS_DIR
from utils.plotter import plot_accuracy, plot_loss


class Classifier:
    def __init__(self):
        self.model = None
        self.history = None
        self.train_generator = None
        self.validation_generator = None
        self.model_name = None
        self.trial_name = None
        self.train_data_dir = absolute_path('split/train')
        self.validation_data_dir = absolute_path('split/val')
        self.test_data_dir = absolute_path('split/test')
        self.classes = 120
        self.epochs = 10
        self.batch_size = 32
        self.img_width, self.img_height = 224, 224
        self.input_shape = (self.img_width, self.img_height, 3)
        self.nb_train_samples = 16418
        self.nb_validation_samples = 4162

    def preprocess_data(self):
        pass

    def train(self):
        pass

    def save_model(self):
        self.model.save_weights(absolute_path('model-{}-{}.h5'.format(self.model_name, self.trial_name)))

    def plot_history(self):
        plot_accuracy(self.history, self.model_name, self.trial_name)
        plot_loss(self.history, self.model_name, self.trial_name)

    def plot_confusion_matrix(self):
        self.validation_generator.reset()
        predictions = self.model.predict_generator(self.validation_generator)
        print(predictions)
        y = np.argmax(predictions, axis=-1)
        print(y)

        print('Classification Report')
        cr = classification_report(y_true=self.validation_generator.classes, y_pred=y)
        print(cr)

        print('Confusion Matrix')
        plt.clf()
        cm = confusion_matrix(self.validation_generator.classes, y)
        df = pd.DataFrame(cm, columns=self.validation_generator.class_indices)
        plt.figure(figsize=(80, 80))
        sn.heatmap(df, annot=True)
        plt.savefig(RESULT_PLOTS_DIR + '/cm-{}-{}.png'.format(self.model_name, self.trial_name))

    def classify_random_images(self):
        pass
