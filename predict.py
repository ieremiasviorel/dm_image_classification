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

from keras.models import load_model


import matplotlib.image as mpimg
from PIL import Image

base_model = NASNetLarge(weights='imagenet', include_top=False)

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(120,activation='softmax')(x) #final layer with softmax activation

model = Model(inputs=base_model.input, outputs=preds)

model.compile(optimizer='Adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

model.load_weights('./models/weights-improvement-naslarge-01.hdf5')
#model = load_model(filepath='./models/weights-improvement-naslarge-01.hdf5')

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

validation_generator = test_datagen.flow_from_directory(
    'split/val',
    target_size=(331, 331),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    shuffle=False)

#validation_generator.reset()
print("start predicts")
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
plt.savefig('./cm_nasnetlarge.png')

imageno=np.random.random_integers(low=0, high=validation_generator.samples)

name = validation_generator.filepaths[imageno]
print(name)
plt.imshow(mpimg.imread(name))

img = Image.open(validation_generator.filepaths[imageno]).resize((331, 331))
probabilities = model.predict(preprocess_input(np.expand_dims(img, axis=0)))
breed_list = tuple(zip(validation_generator.class_indices.values(), validation_generator.class_indices.keys()))

for i in probabilities[0].argsort()[-3:][::-1]: 
    print(probabilities[0][i], "  :  " , breed_list[i])