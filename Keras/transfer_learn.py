# example of using transfer learning where we can select an existing model and add our layers to it
import glob
import matplotlib.pyplot as plt 

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import Model 
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator

# suppresses level 1 and 0 warning messages
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

# Get count of number of files in this folder and subfolders
def get_num_files(path):
    if not os.path.exists(path):
        return 0
    return sum([len(files) for r, d, files in os.walk(path)])

# Get count of number of subfolders directly below the folder in path
def get_num_subfolders(path):
    if not os.path.exists(path):
        return 0
    return sum([len(d) for r, d, files in os.walk(path)])

# image generator function
# this creates new image from old image that is slightly different
# this is to augment the number of images so that we have more images to train
def create_img_generator():
    return ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

Image_width, Image_height = 299, 299
Training_Epochs = 2
Batch_Size = 32
Number_FC_Neurons = 1024

train_dir = './data/train'
validate_dir = './data/validate'

num_train_samples = get_num_files(train_dir)
num_classes=get_num_subfolders(train_dir)
num_validate_samples = get_num_files(validate_dir)

print("num_train_samples : " + str(num_train_samples))
print("num_validate_samples : " + str(num_validate_samples))

num_epoch=Training_Epochs
batch_size = Batch_Size

# Define image generators for training and testing
train_image_gen = create_img_generator()
test_image_gen = create_img_generator()

# connect the image generator to a folder containing the source images the image generator alters.
# Training image generator
train_generator = train_image_gen.flow_from_directory(train_dir, target_size=(Image_width, Image_height), batch_size=batch_size, seed=42)

# Validation image generator
validation_generator = test_image_gen.flow_from_directory(validate_dir, target_size=(Image_width, Image_height), batch_size=batch_size, seed=42)

# Load the inception model
InceptionV3_base_model = InceptionV3(weights='imagenet', include_top=False) #include_top false excludes the final fully connected (FC) layer
print('Inception V3 base model without last FC loaded')

# Define the layers in the new classification prediction
x = InceptionV3_base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(Number_FC_Neurons, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Define trainable model which links input from the Inception V3 base model to new classification prediction layers
model = Model(inputs=InceptionV3_base_model.input, outputs=predictions)

#print(model.summary())

# Option 1: Basic Transfer Learning
print("\nPerforming Transfer Learning")

# Freeze old layers in the Inception base model
Layers_To_Freeze = 172
for layer in model.layers[:Layers_To_Freeze]:
    layer.trainable = False
for layer in model.layers[Layers_To_Freeze:]:
    layer.trainable = True

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# Fit using the image generator
history_fine_tune = model.fit_generator(
  train_generator,
  steps_per_epoch = num_train_samples // batch_size,
  epochs=num_epoch,
  validation_data=validation_generator,
  validation_steps = num_validate_samples // batch_size,
    class_weight='auto')

# save fine tuned model
model.save('inceptionv3_finetuned.model')