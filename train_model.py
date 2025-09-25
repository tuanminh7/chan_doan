import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = 'dataset'
TARGET_SIZE = (320, 320)
BATCH_SIZE = 16
HEAD_EPOCHS = 8
FINE_TUNE_EPOCHS = 30
UNFREEZE_LAYERS = 30
MODEL_SAVE_PATH = 'mo_hinh_best.h5'


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    shear_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    validation_split=0.2
)

val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR, target_size=TARGET_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='training', shuffle=True, seed=SEED
)

val_generator = val_datagen.flow_from_directory(
    DATA_DIR, target_size=TARGET_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='validation', shuffle=False, seed=SEED
)

num_classes = train_generator.num_classes


classes = train_generator.classes
class_weights = compute_class_weight('balanced', classes=np.unique(classes), y=classes)
class_weights = dict(enumerate(class_weights))


base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=['accuracy']
)


checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-7, verbose=1)
earlystop_head = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)


history_head = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=HEAD_EPOCHS,
    class_weight=class_weights,
    callbacks=[earlystop_head, reduce_lr, checkpoint]
)


base_model.trainable = True
for layer in base_model.layers[:-UNFREEZE_LAYERS]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=['accuracy']
)

earlystop_ft = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)

history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=FINE_TUNE_EPOCHS,
    initial_epoch=len(history_head.history['accuracy']),
    class_weight=class_weights,
    callbacks=[earlystop_ft, reduce_lr, checkpoint]
)


def plot_history(h1, h2=None):
    acc = h1.history.get('accuracy', [])
    val_acc = h1.history.get('val_accuracy', [])
    loss = h1.history.get('loss', [])
    val_loss = h1.history.get('val_loss', [])

    if h2:
        acc += h2.history.get('accuracy', [])
        val_acc += h2.history.get('val_accuracy', [])
        loss += h2.history.get('loss', [])
        val_loss += h2.history.get('val_loss', [])

    epochs = range(1, len(acc) + 1)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Train acc')
    plt.plot(epochs, val_acc, label='Val acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Train loss')
    plt.plot(epochs, val_loss, label='Val loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_history(history_head, history_finetune)

from sklearn.metrics import classification_report, confusion_matrix
val_steps = int(np.ceil(val_generator.samples / val_generator.batch_size))
val_generator.reset()
preds = model.predict(val_generator, steps=val_steps)
y_pred = np.argmax(preds, axis=1)
y_true = val_generator.classes[:len(y_pred)]

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=list(train_generator.class_indices.keys())))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print(f"✅ Mô hình tốt nhất đã được lưu tại: {MODEL_SAVE_PATH}")
