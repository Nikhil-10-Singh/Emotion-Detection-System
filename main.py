
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Set paths to training and testing directories
train_dir = 'C:\\Users\\Lenovo!\\OneDrive\\Desktop\\Project\\emotion\\dataset\\train'
test_dir = 'C:\\Users\\Lenovo!\\OneDrive\\Desktop\\Project\\emotion\\dataset\\test'

# Define image dimensions, batch size, and number of classes
img_width, img_height = 224, 224
batch_size = 32
num_classes = 7  # Number of emotion classes (e.g., angry, h
# appy, sad, surprise, neutral)

# Use ImageDataGenerator to load and preprocess the images
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training and test data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# Define a new CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model on the training data, validate on the test data
epochs = 15  # You can increase the number of epochs for better accuracy
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=test_generator
)

# Save the trained model
model.save("trained_emotion_model.h5")

# Evaluate model accuracy on the test set
loss, accuracy = model.evaluate(test_generator)
print(f"Model accuracy on test set: {accuracy * 100:.2f}%")

# (Optional) Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
