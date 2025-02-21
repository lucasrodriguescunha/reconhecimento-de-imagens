import cv2
import numpy as np
import os

def carregar_e_preparar_imagem(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Erro ao carregar a imagem: {image_path}")

    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def processar_dataset(dataset_path):
    imagens = []
    labels = []

    for classe in os.listdir(dataset_path):
        classe_path = os.path.join(dataset_path, classe)
        if os.path.isdir(classe_path):
            for imagem in os.listdir(classe_path):
                imagem_path = os.path.join(classe_path, imagem)
                try:
                    img = carregar_e_preparar_imagem(imagem_path)
                    imagens.append(img)
                    labels.append(1 if classe == "aprovadas" else 0)
                except:
                    continue

    return np.array(imagens), np.array(labels)

if __name__ == "__main__":
    dataset_path = "dataset/"
    processar_dataset(dataset_path)
