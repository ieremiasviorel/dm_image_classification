from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

from classifier import Classifier


class SimpleCNN(Classifier):
    def __init__(self):
        super(SimpleCNN, self).__init__()

    def train(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, (2, 2), input_shape=self.input_shape))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(32, (2, 2)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (2, 2)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(120))
        self.model.add(Activation('sigmoid'))

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='Adam',
            metrics=['accuracy'])

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            self.train_data_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size, class_mode='categorical')

        validation_generator = test_datagen.flow_from_directory(
            self.validation_data_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size, class_mode='categorical')

        self.history = self.model.fit_generator(
            train_generator,
            validation_data=validation_generator,
            epochs=self.epochs,
            steps_per_epoch=self.nb_train_samples // self.batch_size,
            validation_steps=self.nb_validation_samples // self.batch_size)
