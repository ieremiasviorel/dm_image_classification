import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

from classification.classifier import Classifier


class VGG16NetClassifier(Classifier):
    def __init__(self):
        super(VGG16NetClassifier, self).__init__()
        self.model_name = 'vgg_16'
        self.trial_name = '01'

    def train(self):
        base_model = VGG16(
            weights='imagenet',
            include_top=False)

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(512, activation='relu')(x)

        preds = Dense(self.classes, activation='softmax')(x)

        self.model = Model(inputs=base_model.input, outputs=preds)

        # specify the inputs
        # specify the outputs
        # now a model has been created based on our architecture
        for i, layer in enumerate(self.model.layers):
            print(i, layer.name)

        for layer in base_model.layers:
            layer.trainable = False

        train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

        self.train_generator = train_datagen.flow_from_directory(
            self.train_data_dir,
            target_size=(self.img_width, self.img_height),
            color_mode='rgb',
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True)

        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

        self.validation_generator = test_datagen.flow_from_directory(
            self.validation_data_dir,
            target_size=(self.img_width, self.img_height),
            color_mode='rgb',
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True)

        # Adam optimizer
        # loss function - categorical cross entropy
        # evaluation metric - accuracy
        self.model.compile(
            optimizer='Adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        step_size_train = self.train_generator.n // self.train_generator.batch_size

        step_size_validation = self.validation_generator.n // self.validation_generator.batch_size

        self.history = self.model.fit_generator(
            generator=self.train_generator,
            steps_per_epoch=step_size_train,
            validation_data=self.validation_generator,
            validation_steps=step_size_validation,
            epochs=self.epochs)

    def classify_random_images(self):
        imageno = np.random.random_integers(low=0, high=self.validation_generator.samples)

        name = self.validation_generator.filepaths[imageno]
        print(name)
        plt.imshow(mpimg.imread(name))

        img = Image.open(self.validation_generator.filepaths[imageno]).resize((self.img_width, self.img_height))
        probabilities = self.model.predict(preprocess_input(np.expand_dims(img, axis=0)))
        breed_list = tuple(
            zip(self.validation_generator.class_indices.values(),
                self.validation_generator.class_indices.keys()))

        for i in probabilities[0].argsort()[-3:][::-1]:
            print(probabilities[0][i], "  :  ", breed_list[i])
