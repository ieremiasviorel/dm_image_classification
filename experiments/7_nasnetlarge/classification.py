import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications.nasnet import NASNetLarge, preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from keras.models import save_model
from keras.callbacks import ModelCheckpoint

import pandas as pd  
import seaborn as sn 
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.image as mpimg
from PIL import Image


PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(''))

IMAGES_DIR = PROJECT_ROOT_DIR + '/input/stanford-dogs-dataset/images/Images'
ANNOTATIONS_DIR = PROJECT_ROOT_DIR + '/input/stanford-dogs-dataset/annotations/Annotation'

RESULT_PLOTS_DIR = PROJECT_ROOT_DIR + '/plots'


def absolute_path(relative_path):
    return PROJECT_ROOT_DIR + '/' + relative_path

train_data_dir = '/content/gdrive/My Drive/dm_image_classification/split/split/train' #absolute_path('split/train')
validation_data_dir = '/content/gdrive/My Drive/dm_image_classification/split/split/val' #absolute_path('split/val')


base_model = NASNetLarge(weights='imagenet', include_top=False)

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(120,activation='softmax')(x) #final layer with softmax activation

model = Model(inputs=base_model.input, outputs=preds)
# specify the inputs
# specify the outputs
# now a model has been created based on our architecture

for i, layer in enumerate(model.layers):
    print(i, layer.name)

for layer in base_model.layers:
    layer.trainable = False

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input)  # included in our dependencies

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(331, 331),
                                                    color_mode='rgb',
                                                    batch_size=32,
                                                    class_mode='categorical',
                                                    shuffle=True)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(331, 331),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    shuffle=False)

model.compile(optimizer='Adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy

step_size_train = train_generator.n//train_generator.batch_size
step_size_validation = validation_generator.n//validation_generator.batch_size

## checkpoints
filepath="/content/gdrive/My Drive/dm_image_classification/weights-improvement-naslarge-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='auto')
callbacks_list = [checkpoint]

model.fit_generator(generator=train_generator,
                    steps_per_epoch=step_size_train,
                    validation_data=validation_generator,
                    validation_steps=step_size_validation,
                    epochs=10,
                    callbacks=callbacks_list)

model.save_weights(filepath='/content/gdrive/My Drive/dm_image_classification/nasnetlarge.h5')
############################################### https://www.kaggle.com/devang/transfer-learning-with-keras-and-mobilenet-v2#Confusion-matrix
# Classification report + confusion matrix 

validation_generator.reset()
predictions = model.predict_generator(validation_generator)
print(predictions)
classes = validation_generator.classes[validation_generator.index_array]
y = np.argmax(predictions, axis=-1)
print(y)

print('Classification Report')
cr = classification_report(y_true=validation_generator.classes, y_pred=y)
print(cr)

print('Confusion Matrix')
plt.clf()
cm = confusion_matrix(validation_generator.classes, y)
df = pd.DataFrame(cm, columns=validation_generator.class_indices)
plt.figure(figsize=(80,80))
sn.heatmap(df, annot=True)
plt.savefig('/content/gdrive/My Drive/dm_image_classification/cm_nasnetlarge.png')