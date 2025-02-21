import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Carregar modelo treinado
model = load_model("models/modelo_controle_qualidade.h5")

def fazer_previsao(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Erro ao carregar a imagem.")
        return None

    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    predicao = model.predict(img)[0][0]
    resultado = "Imagem Aprovada" if predicao > 0.5 else "Imagem Rejeitada"
    return resultado

if __name__ == "__main__":
    imagem_teste = "dataset/teste.jpg"
    print(fazer_previsao(imagem_teste))
