import matplotlib.pyplot as plt

from definitions import RESULT_PLOTS_DIR


def plot_accuracy(history, model_name, trial_name):
    _config_plot('accuracy', history.params.get('epochs'))
    plt.title('{} accuracy'.format(model_name))
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(RESULT_PLOTS_DIR + '/accuracy_{}_{}.png'.format(model_name, trial_name))


def plot_loss(history, model_name, trial_name):
    _config_plot('loss', history.params.get('epochs'))
    plt.title('{} loss'.format(model_name))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(RESULT_PLOTS_DIR + '/loss_{}_{}.png'.format(model_name, trial_name))


def _config_plot(plot_type, x_lim):
    plt.clf()
    plt.autoscale(False)

    plt.figure(figsize=(6.4, 4.8))

    plt.xlim(0, x_lim)
    if plot_type == 'accuracy':
        plt.ylim(0.0, 1.0)
    plt.xlabel('epoch')
    plt.ylabel(plot_type)

    ax = plt.gca()

    ax.grid(which='major')
    ax.grid(which='minor', alpha=0.3)
