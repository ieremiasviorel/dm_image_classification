from definitions import absolute_path
from utils.plotter import plot_accuracy, plot_loss


class Classifier:
    def __init__(self):
        self.model = None
        self.history = None
        self.train_data_dir = absolute_path('../split/train')
        self.validation_data_dir = absolute_path('../split/val')
        self.test_data_dir = absolute_path('split/test')
        self.epochs = 10
        self.batch_size = 32
        self.img_width, self.img_height = 224, 224
        self.input_shape = (self.img_width, self.img_height, 3)
        self.nb_train_samples = 16418
        self.nb_validation_samples = 4162
        self.model_name = None
        self.trial_name = None

    def preprocess_data(self):
        pass

    def train(self):
        pass

    def save_model(self):
        self.model.save_weights(absolute_path('model-{}-{}.h5'.format(self.model_name, self.trial_name)))

    def plot_history(self):
        plot_accuracy(self.history, self.model_name, self.trial_name)
        plot_loss(self.history, self.model_name, self.trial_name)
