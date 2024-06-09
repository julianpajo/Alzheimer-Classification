import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import RMSprop, Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from imblearn.over_sampling import SMOTE
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import BatchNormalization

# Set directories
train_dir = '../dataset/train'
test_dir = '../dataset/test'

labels = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]

image_counts = {}
for label in labels:
    label_dir = os.path.join(train_dir, label)
    count = len([file for file in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, file))])
    image_counts[label] = count

# Creare un DataFrame per i dati
df = pd.DataFrame(list(image_counts.items()), columns=['label', 'count'])

# Creare il grafico per contare il numero di immagini per categoria
plt.figure(figsize=(15, 8))
ax = sns.barplot(x='label', y='count', data=df, palette='Set1')
ax.set_xlabel("Class", fontsize=20)
ax.set_ylabel("Count", fontsize=20)
plt.title('The Number Of Samples For Each Class', fontsize=20)
plt.grid(True)

# Salvare il grafico
plt.savefig('res/num_images_per_category.png')

# Get class names
class_names = os.listdir(train_dir)

# Print sample images from each class
plt.figure(figsize=(5, 5))
for i, class_name in enumerate(class_names):
    # Get a random image from the class directory
    img_name = np.random.choice(os.listdir(os.path.join(train_dir, class_name)))
    img_path = os.path.join(train_dir, class_name, img_name)

    # Read and display the image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(2, 2, i + 1)
    plt.imshow(img)
    plt.title(class_name)
    plt.axis('off')

plt.savefig('res/image_per_category.png')

# Load image paths and labels
images = []
labels = []
for class_name in class_names:
    class_dir = os.path.join(train_dir, class_name)
    for img_name in os.listdir(class_dir):
        images.append(os.path.join(class_dir, img_name))
        labels.append(class_name)
df = pd.DataFrame({'image': images, 'label': labels})

Size = (128, 128)

# Data Augmentation
work_dr = ImageDataGenerator(
    rescale=1. / 255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data_gen = work_dr.flow_from_dataframe(df, x_col='image', y_col='label', target_size=Size, batch_size=6500,
                                             shuffle=False)

# Load the test data
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=Size,
    batch_size=32,
    class_mode='categorical'
)

for i in range(len(train_data_gen)):
    train_data, train_labels = train_data_gen[i]

class_num = np.sort(['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented'])

# Apply SMOTE to balance the dataset
sm = SMOTE(random_state=42)
train_data, train_labels = sm.fit_resample(train_data.reshape(-1, 128 * 128 * 3), train_labels)
train_data = train_data.reshape(-1, 128, 128, 3)
print(train_data.shape, train_labels.shape)

labels = [class_num[i] for i in np.argmax(train_labels, axis=1)]
plt.figure(figsize=(15, 8))
ax = sns.countplot(x=labels, palette='Set1')
ax.set_xlabel("Class", fontsize=20)
ax.set_ylabel("Count", fontsize=20)
plt.title('The Number Of Samples For Each Class', fontsize=20)
plt.grid(True)
plt.savefig('res/image_per_category_smote.png')


# Split the data into training, validation, and test sets
X_train, X_test1, y_train, y_test1 = train_test_split(train_data, train_labels, test_size=0.3, random_state=42,
                                                      shuffle=True, stratify=train_labels)
X_val, X_test, y_val, y_test = train_test_split(X_test1, y_test1, test_size=0.5, random_state=42, shuffle=True,
                                                stratify=y_test1)

vgg_base = VGG19(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Set the first 15 layers to be non-trainable
for layer in vgg_base.layers:
    layer.trainable = False

# Set the last 5 layers to be trainable
for layer in vgg_base.layers[-5:]:
    layer.trainable = True

# Add custom layers
x = Flatten()(vgg_base.output)
x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
output = Dense(4, activation='softmax', kernel_regularizer=l2(0.001))(x)


model = Model(inputs=vgg_base.input, outputs=output)

model.summary()

# Set the callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Set the learning rate schedule
initial_learning_rate = 0.001
lr_schedule = ExponentialDecay(
    initial_learning_rate,
    decay_steps=10000,
    decay_rate=0.9,
    staircase=True
)

optimizer = Adam(learning_rate=lr_schedule)
batch_size = 32

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

hist = model.fit(X_train, y_train, epochs=50, batch_size=batch_size, validation_data=(X_val, y_val),
                 callbacks=[early_stopping])

model.save('res/modello_cnr_alzheimer.h5')

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Generate predictions
test_predictions = model.predict(test_generator)
predicted_classes = np.argmax(test_predictions, axis=1)

# True classes
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Accuracy
accuracy = accuracy_score(true_classes, predicted_classes)
print(accuracy)

# Classification Report
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

# Plot the training history
hist_ = pd.DataFrame(hist.history)

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.plot(hist_['loss'], label='Train_Loss')
plt.plot(hist_['val_loss'], label='Validation_Loss')
plt.title('Train_Loss & Validation_Loss', fontsize=20)
plt.xlabel('Epochs', fontsize=16)  # Label per l'asse x
plt.ylabel('Loss', fontsize=16)  # Label per l'asse y
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(hist_['accuracy'], label='Train_Accuracy')
plt.plot(hist_['val_accuracy'], label='Validation_Accuracy')
plt.title('Train_Accuracy & Validation_Accuracy', fontsize=20)
plt.xlabel('Epochs', fontsize=16)  # Label per l'asse x
plt.ylabel('Accuracy', fontsize=16)  # Label per l'asse y
plt.legend()
plt.savefig('res/train_loss_accuracy.png')
