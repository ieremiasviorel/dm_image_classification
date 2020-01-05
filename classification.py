import numpy as np
import os
import keras
import matplotlib.pyplot as plt
<<<<<<< HEAD
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import MobileNet
=======
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications.vgg16 import VGG16,preprocess_input
>>>>>>> Fixed bug in classification model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam

base_model = VGG16(include_top=False,
                  input_shape = (224,224,3),
                  weights = 'imagenet')

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(3,activation='softmax')(x) #final layer with softmax activation

model = Model(inputs=base_model.input, outputs=preds)
# specify the inputs
# specify the outputs
# now a model has been created based on our architecture

for i, layer in enumerate(model.layers):
    print(i, layer.name)

for layer in base_model.layers:
    layer.trainable=False

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input)  # included in our dependencies

train_generator = train_datagen.flow_from_directory('./split/train',
                                                    target_size=(224, 224),
                                                    color_mode='rgb',
                                                    batch_size=32,
                                                    class_mode='categorical',
                                                    shuffle=True)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

validation_generator = test_datagen.flow_from_directory(
    './split/val',
    target_size=(224, 224),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    shuffle=True)

model.compile(optimizer='Adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy

step_size_train = train_generator.n//train_generator.batch_size
step_size_validation = validation_generator.n//validation_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=step_size_train,
                    validation_data=validation_generator,
                    validation_steps=step_size_validation,
                    epochs=10)
