import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os


def treinar_modelo():
    dataset_path = "dataset/"

    if not os.path.exists(dataset_path):
        print("Erro: O diretório do dataset não foi encontrado.")
        return

    # Pré-processamento e Data Augmentation
    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    batch_size = 32
    img_size = (224, 224)

    train_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    val_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    # Definição do Modelo CNN
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Saída binária (1 ou 0)
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Treinamento
    model.fit(train_generator, validation_data=val_generator, epochs=10)

    # Salvar modelo treinado
    os.makedirs("models", exist_ok=True)
    model.save("models/modelo_controle_qualidade.h5")
    print("Modelo treinado e salvo com sucesso!")


if __name__ == "__main__":
    treinar_modelo()
