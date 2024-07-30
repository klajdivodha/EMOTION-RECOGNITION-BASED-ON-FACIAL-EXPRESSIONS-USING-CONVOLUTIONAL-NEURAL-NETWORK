import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam

train_dir = r'C:\Users\User\Desktop\FRE-2013\train'
test_dir = r'C:\Users\User\Desktop\FRE-2013\test'

ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

# Set up data generators
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, zoom_range=0.2)
val_datagen = ImageDataGenerator(rescale=1./255)

# Load and preprocess data
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(48, 48),
                                                    batch_size=64,
                                                    color_mode='grayscale',
                                                    class_mode='categorical')

val_generator = val_datagen.flow_from_directory(test_dir,
                                                target_size=(48, 48),
                                                batch_size=64,
                                                color_mode='grayscale',
                                                class_mode='categorical')

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1)

# Train the model
model.fit(train_generator,
          steps_per_epoch=train_generator.samples // train_generator.batch_size,
          epochs=50,
          validation_data=val_generator,
          validation_steps=val_generator.samples // val_generator.batch_size,
          callbacks=[early_stop])

# Evaluate the model on the test set
test_generator = val_datagen.flow_from_directory(test_dir,
                                                 target_size=(48, 48),
                                                 batch_size=64,
                                                 color_mode='grayscale',
                                                 class_mode='categorical')

test_loss, test_acc = model.evaluate(test_generator)
print('Test accuracy:', test_acc)

# Save the final model
tf.keras.models.save_model(model, 'final_model1.keras')