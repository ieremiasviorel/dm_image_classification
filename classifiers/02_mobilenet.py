from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

from classifier import Classifier


class MobileNetClassifier(Classifier):
    def __init__(self):
        super(MobileNetClassifier, self).__init__()

    def train(self):
        # imports the mobilenet model and discards the last 1000 neuron layer.
        base_model = MobileNet(weights='imagenet', include_top=False)

        x = base_model.output

        x = GlobalAveragePooling2D()(x)
        # add dense layers so that the model can learn more complex functions and classify for better results
        x = Dense(1024, activation='relu')(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        # final layer with SoftMax activation
        preds = Dense(120, activation='softmax')(x)

        self.model = Model(inputs=base_model.input, outputs=preds)

        # specify the inputs
        # specify the outputs
        # now a model has been created based on our architecture
        for i, layer in enumerate(self.model.layers):
            print(i, layer.name)

        for layer in self.model.layers:
            layer.trainable = False

        train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

        train_generator = train_datagen.flow_from_directory(
            './data',
            target_size=(self.img_width, self.img_height),
            color_mode='rgb',
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True)

        # Adam optimizer
        # loss function - categorical cross entropy
        # evaluation metric - accuracy
        self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

        step_size_train = train_generator.n // train_generator.batch_size

        self.model.fit_generator(
            generator=train_generator,
            steps_per_epoch=step_size_train,
            epochs=10)