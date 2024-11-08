import matplotlib.pyplot as plt
from keras.api.layers import GlobalAveragePooling2D
from keras.api.applications import VGG19, ResNet50
from keras.api.layers import Conv2D, MaxPooling2D
from keras.api import Sequential
from keras.api.layers import Dense, Flatten, Input
from keras.src.legacy.preprocessing.image import ImageDataGenerator

IMG_SIZE = (64, 64)
IMG_SHAPE = IMG_SIZE + (3,)
BATCH_SIZE = 16
epochs = 5


def plot_learning_curves(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    # plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='lower right')
    plt.ylabel('Loss')
    # plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')

    plt.show()


def plot_first_n_images(generator, n):
    images, labels = next(generator)

    plt.figure(figsize=(10, 5))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i])
        plt.title('Dog' if labels[i] == 1 else 'Cat')
        plt.axis('off')
    plt.show()


def create_fully_connected_model():
    model = Sequential([
        Input(shape=IMG_SHAPE),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def create_cnn_model():
    model = Sequential([
        Input(shape=IMG_SHAPE),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def create_vgg19_model():
    base_model = VGG19(weights='imagenet', include_top=False)
    for layer in base_model.layers:
        layer.trainable = False
    print('\nBase model VGG19')
    print(base_model.summary())
    model = Sequential([
        Input(shape=IMG_SHAPE),
        base_model,
        GlobalAveragePooling2D(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    print('\nModel VGG19 with a new classification head')
    print(model.summary())
    print('\nModel VGG19 trainable var len')
    print(len(model.trainable_variables))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def create_resnet_model():
    base_model = ResNet50(weights='imagenet', include_top=False)
    for layer in base_model.layers:
        layer.trainable = False
    print('\nBase model ResNet')
    print(base_model.summary())
    model = Sequential([
        Input(shape=IMG_SHAPE),
        base_model,
        GlobalAveragePooling2D(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    print('\nModel ResNet with a new classification head')
    print(model.summary())
    print('\nModel ResNet trainable var len')
    print(len(model.trainable_variables))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

print('Train dataset:')
train_generator = train_datagen.flow_from_directory(
    'Cats_and_Dogs/train/',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

print('Validation dataset:')
validation_generator = train_datagen.flow_from_directory(
    'Cats_and_Dogs/val/',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
)

print('Test dataset:')
test_generator = train_datagen.flow_from_directory(
    'Cats_and_Dogs/test/',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# plot_first_n_images(train_generator, 5)
# plot_first_n_images(validation_generator, 5)
# plot_first_n_images(test_generator, 5)

# model_a = create_fully_connected_model()
#
# history = model_a.fit(
#     train_generator,
#     validation_data=validation_generator,
#     epochs=epochs
# )
#
# test_loss, test_acc = model_a.evaluate(test_generator, verbose=2)
# print('test_lost: ', test_loss, ' test_acc: ', test_acc)
# plot_learning_curves(history)
# model_b = create_cnn_model()
#
# history = model_b.fit(
#     train_generator,
#     validation_data=validation_generator,
#     epochs=epochs
# )
#
# test_loss, test_acc = model_b.evaluate(test_generator, verbose=2)
# print('test_lost: ', test_loss, ' test_acc: ', test_acc)
# plot_learning_curves(history)
# model_vgg19 = create_vgg19_model()
#
# history = model_vgg19.fit(
#     train_generator,
#     validation_data=validation_generator,
#     epochs=epochs
# )
#
# test_loss, test_acc = model_vgg19.evaluate(test_generator, verbose=2)
# print('test_lost: ', test_loss, ' test_acc: ', test_acc)
# plot_learning_curves(history)
# model_resnet = create_resnet_model()
#
# history = model_resnet.fit(
#     train_generator,
#     validation_data=validation_generator,
#     epochs=epochs
# )
#
# test_loss, test_acc = model_resnet.evaluate(test_generator, verbose=2)
# print('test_lost: ', test_loss, ' test_acc: ', test_acc)
# plot_learning_curves(history)
