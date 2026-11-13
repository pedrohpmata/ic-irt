import os
import cv2  # Faz leitura, filtros e gravação de imagens
import numpy as np

# Devolve a imagem com os defeitos em branco e o resto em preto
def normalizar(imagem):
    # Estima o fundo com um borrão forte
    fundo = cv2.GaussianBlur(imagem, (0, 0), 20)
    #Retira o fundo
    diferenca = imagem - fundo
    # Borra um pouco para reduzir o ruído sem apagar os defeitos
    diferenca = cv2.GaussianBlur(diferenca, (0, 0), 2)
    # Normaliza os pixels
    z = (diferenca - diferenca.mean()) / diferenca.std()
    # Aumenta o contraste
    z = np.clip((z - 1.5) / (3 - 1.5), 0, 1)
    # Volta para a escala 0-255 de imagem
    return (z * 255).astype(np.uint8)


# IoU = interseção / união entre a saída e a máscara (0 a 1)
def calcular_iou(previsto, esperado):
    intersecao = np.logical_and(previsto, esperado).sum()
    uniao = np.logical_or(previsto, esperado).sum()
    # Duas máscaras vazias contam como acerto (evita 0/0)
    if uniao == 0:
        return 1.0
    return intersecao / uniao


# Caminhos das pastas
pasta_programa = os.path.dirname(os.path.abspath(__file__))
pasta_dataset = os.path.join(pasta_programa, "Thermal Inspection Dataset for Defect Segmentation in CFRP Laminates")
pasta_originais = os.path.join(pasta_dataset, "originalData")  # Imagens térmicas
pasta_mascaras = os.path.join(pasta_dataset, "annotatedData")  # Máscaras com o mesmo nome das originais
pasta_saida = os.path.join(pasta_dataset, "annotatedDataGaussianBlur")  # Imagens geradas
# Cria a pasta de saída
os.makedirs(pasta_saida, exist_ok=True)
# Lista os .png em ordem numérica
nomes = sorted(
    [n for n in os.listdir(pasta_originais) if n.lower().endswith(".png")],
    key=lambda n: int(n.split(".")[0]),
)
ious = []  # IoU de todas as imagens
ious_com_defeito = []  # IoU só das imagens com defeito na máscara
linhas = []  # Linhas do iou.txt
sem_mascara = []  # Imagens sem máscara, ficam fora do IoU
for nome in nomes:
    # Abre a imagem em tons de cinza
    imagem = cv2.imread(os.path.join(pasta_originais, nome), 0)
    # Pula se não conseguiu abrir
    if imagem is None:
        print(nome, "-> nao foi possivel abrir a imagem")
        continue
    imagem = imagem.astype(np.float32)
    resultado = normalizar(imagem)
    # Salva a saída
    cv2.imwrite(os.path.join(pasta_saida, nome), resultado)
    caminho_mascara = os.path.join(pasta_mascaras, nome)
    # Pula as imagens sem máscara
    if not os.path.exists(caminho_mascara):
        sem_mascara.append(nome)
        continue
    mascara = cv2.imread(caminho_mascara, 0)
    # Branco (acima de 127) = defeito
    previsto = resultado > 127
    esperado = mascara > 127
    iou = calcular_iou(previsto, esperado)
    ious.append(iou)
    if esperado.any():  # Máscara com defeito
        ious_com_defeito.append(iou)
    linhas.append(f"{nome} IoU: {iou:.4f} (mascara com defeito: {'sim' if esperado.any() else 'nao'})")
    print(nome, "-> IoU:", f"{iou:.4f}")
# Resumo no final do iou.txt
linhas.append("")
linhas.append(f"imagens comparadas: {len(ious)}")
if len(ious) > 0:
    linhas.append(f"IoU medio (todas): {np.mean(ious):.4f}")
    # Média só das imagens com defeito (máscaras vazias distorcem a média geral)
    if len(ious_com_defeito) > 0:
        linhas.append(f"IoU medio (so as que tem defeito): {np.mean(ious_com_defeito):.4f} ({len(ious_com_defeito)} imagens)")
if len(sem_mascara) > 0:
    linhas.append(f"imagens sem mascara (fora da media): {len(sem_mascara)}")
with open(os.path.join(pasta_saida, "iou.txt"), "w", encoding="utf-8") as arquivo:
    arquivo.write("\n".join(linhas))
# Mostra as médias na tela
print()
print("\n".join(linhas[-5:]))