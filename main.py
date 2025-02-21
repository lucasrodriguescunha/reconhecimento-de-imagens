import argparse
from src.train_model import treinar_modelo
from src.predict import fazer_previsao
from src.utils import verificar_estrutura_pastas

def main():
    parser = argparse.ArgumentParser(description="Sistema de Reconhecimento de Imagens para Controle de Qualidade")
    parser.add_argument("--train", action="store_true", help="Treinar o modelo")
    parser.add_argument("--predict", type=str, help="Fazer previsão em uma imagem (caminho do arquivo)")

    args = parser.parse_args()

    verificar_estrutura_pastas()

    if args.train:
        treinar_modelo()
    elif args.predict:
        resultado = fazer_previsao(args.predict)
        print("Resultado da previsão:", resultado)
    else:
        print("Use --train para treinar o modelo ou --predict <caminho_da_imagem> para fazer uma previsão.")

if __name__ == "__main__":
    main()
