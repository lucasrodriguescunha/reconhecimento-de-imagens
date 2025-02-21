import os
import cv2


def verificar_estrutura_pastas():
    """Verifica se os diretórios necessários existem e os cria se necessário."""
    pastas = ["dataset", "models"]
    for pasta in pastas:
        if not os.path.exists(pasta):
            os.makedirs(pasta)
            print(f"Criado diretório: {pasta}")


def exibir_imagem(image_path):
    """Exibe uma imagem na tela."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Erro ao carregar imagem: {image_path}")
        return
    cv2.imshow("Imagem", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
