from utils.plotter import plot_accuracy, plot_loss


class Classifier:
    def __init__(self):
        self.model = None
        self.history = None
        self.train_data_dir = 'split/train'
        self.validation_data_dir = 'split/val'
        self.epochs = 10
        self.batch_size = 16
        self.img_width, self.img_height = 224, 224
        self.nb_train_samples = 16418
        self.nb_validation_samples = 4162

    def preprocess_data(self):
        pass

    def train(self):
        pass

    def save_model(self):
        self.model.save_weights('model_saved.h5')

    def plot_history(self, model_name, trial_name):
        plot_accuracy(self.history, model_name, trial_name)
        plot_loss(self.history, model_name, trial_name)
