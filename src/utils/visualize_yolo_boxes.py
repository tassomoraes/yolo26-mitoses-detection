"""
Script para visualizar imagens com bounding boxes YOLO de forma iterativa.

Uso:
    python visualize_yolo_boxes.py

O script solicitará o caminho da pasta contendo as imagens e seus labels.
Pressionando qualquer tecla, passa para a próxima imagem.
"""

import os
import sys
from pathlib import Path
import cv2


def draw_yolo_check(image_path, label_path, output_path):
    """
    Desenha bounding boxes YOLO sobre a imagem e exibe.
    
    Args:
        image_path: caminho para a imagem
        label_path: caminho para o arquivo de labels YOLO
    """
    # Carrega a imagem
    img = cv2.imread(image_path)
    if img is None:
        print(f"Erro ao carregar imagem: {image_path}")
        return
    
    h, w, _ = img.shape

    # Lê o arquivo de label, se existir
    if not os.path.exists(label_path):
        print(f"Arquivo de label não encontrado: {label_path}")
        # Mostra imagem sem boxes
        cv2.imshow("Verificacao YOLO", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    with open(label_path, 'r') as f:
        lines = f.readlines()

    # Desenha cada bounding box
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
            
        cls = parts[0]
        x_c, y_c, bw, bh = map(float, parts[1:])

        # Converte de coordenadas normalizadas para pixels
        xmin = int((x_c - bw / 2) * w)
        ymin = int((y_c - bh / 2) * h)
        xmax = int((x_c + bw / 2) * w)
        ymax = int((y_c + bh / 2) * h)

        # Desenha o retângulo (Verde para classe 0, Vermelho para classe 1)
        if cls == "0":
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(img, f"Classe: {cls}", (xmin, ymin - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            cv2.putText(img, f"Classe: {cls}", (xmin, ymin - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Salva imagem com as bounding boxes
    file_name = image_path.split('/')[-1]
    cv2.imwrite(f'{output_path}/{file_name}', img)
    print(f"Imagem salva em: {output_path}/{file_name}")
    
    # Mostra o resultado
    cv2.imshow("Verificacao YOLO", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_label_path(image_path, labels_folder):
    """
    Encontra o arquivo de label correspondente à imagem.
    
    Args:
        image_path: caminho da imagem
        labels_folder: pasta contendo os labels
    
    Returns:
        caminho do arquivo de label ou None se não encontrado
    """
    image_name = Path(image_path).stem  # nome sem extensão
    print(f"Procurando label para '{image_name}' em '{labels_folder}'...")
    
    # Procura por arquivo de label com o mesmo nome mas com extensão .txt
    label_path = os.path.join(labels_folder, f"{image_name}.txt")
    
    if os.path.exists(label_path):
        return label_path
    
    return None


def visualize_folder(folder_path, output_path):
    """
    Visualiza todas as imagens de uma pasta com seus bounding boxes.
    
    Args:
        folder_path: caminho da pasta contendo imagens
        labels_path: caminho da pasta contendo os labels
    """
    labels_path = ""
    labels_path = folder_path.replace("images", "labels") if not labels_path else labels_path
    folder_path = Path(folder_path)
    output_path = Path(output_path)
    labels_folder = Path(labels_path)

    if not output_path.exists():
        os.makedirs(output_path)

    if not folder_path.exists():
        print(f"Erro: Pasta '{folder_path}' não existe!")
        sys.exit(1)
    
    # Formatos de imagem suportados
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
    
    # Encontra todas as imagens
    images = []
    for ext in image_extensions:
        images.extend(folder_path.glob(f"*{ext}"))
        images.extend(folder_path.glob(f"*{ext.upper()}"))
    
    images = sorted(list(set(images)))  # Remove duplicatas e ordena
    
    if not images:
        print(f"Nenhuma imagem encontrada em '{folder_path}'")
        sys.exit(1)
    
    print(f"Encontradas {len(images)} imagem(ns)")


    # altera o nome 'images' em folder path para 'labels'
    # labels_folder = folder_path.parent 
    # labels_folder = labels_folder.parent / "labels" / str(folder_path.name).split("/")[-1]
    
    if labels_folder.exists() and labels_folder.is_dir():
        print(f"Encontrada pasta de labels: '{labels_folder}'")
    else:
        print(f"Não encontrada pasta de labels em '{labels_folder}'")
        labels_folder = None
        

    # Exibe cada imagem
    for i, image_path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] {image_path.name}")
        
        label_path = None
        if labels_folder:
            label_path = get_label_path(str(image_path), str(labels_folder))
        
        if label_path:
            print(f"  Label: {os.path.basename(label_path)}")
        else:
            print("  Sem label encontrado")
        
        draw_yolo_check(str(image_path), label_path if label_path else "", output_path)
    
    print("\nVisualizacao concluida!")


def main():
    """Função principal."""
    print("=== Visualizador de YOLO Bounding Boxes ===\n")
    
    # Solicita o caminho da pasta
    folder_path = "test/images" # input("Digite o caminho da pasta contendo as imagens: ").strip()
    output_path = "checkbox_images" # input("Digite o caminho da pasta de saída: ").strip()

    if not folder_path:
        print("Erro: Caminho não fornecido!")
        sys.exit(1)
    

    visualize_folder(folder_path, output_path)


if __name__ == "__main__":
    main()
