import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras.applications.nasnet import NASNetLarge, preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

from definitions import absolute_path

base_model = NASNetLarge(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(
    x)  # we add dense layers so that the model can learn more complex functions and classify for better results.
x = Dense(1024, activation='relu')(x)  # dense layer 2
x = Dense(512, activation='relu')(x)  # dense layer 3
preds = Dense(120, activation='softmax')(x)  # final layer with softmax activation

model = Model(inputs=base_model.input, outputs=preds)

model.load_weights(absolute_path('weights-improvement-naslarge-01.hdf5'))

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

validation_generator = test_datagen.flow_from_directory(
    './split/val',
    target_size=(331, 331),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    shuffle=False)

imageno = np.random.random_integers(low=0, high=validation_generator.samples)

name = validation_generator.filepaths[imageno]
print(name)
plt.imshow(mpimg.imread(name))

img = Image.open(validation_generator.filepaths[imageno]).resize((224, 224))
probabilities = model.predict(preprocess_input(np.expand_dims(img, axis=0)))
breed_list = tuple(zip(validation_generator.class_indices.values(), validation_generator.class_indices.keys()))

for i in probabilities[0].argsort()[-3:][::-1]:
    print(probabilities[0][i], "  :  ", breed_list[i])

print(breed_list)
