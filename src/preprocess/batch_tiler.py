"""
Script para processar em lote todas as imagens e criar tiles com labels.

Processa:
  1. Lê um CSV (datasets_xvalidation.csv) que mapeia cada imagem para seu split
  2. Itera sobre todas as imagens na pasta de entrada
  3. Para cada imagem, obtém o split correspondente do CSV
  4. Cria os tiles usando tile_image_with_labels()
  5. Salva os tiles na estrutura de pastas: saida/images/{split} e saida/labels/{split}

Estrutura esperada:
  - images/: pasta com imagens de entrada (001.tiff, 002.tiff, ...)
  - labels/: pasta com labels YOLO correspondentes (001.txt, 002.txt, ...)
  - data_split.csv: CSV com colunas "Slide" e "Dataset"

Saída:
  - saida/images/train/, saida/images/test/, etc: tiles agrupados por split
  - saida/labels/train/, saida/labels/test/, etc: labels agrupados por split
"""

import os
import pandas as pd
from pathlib import Path
from tile_label_creator import tile_image_with_labels


# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================

# Pastas de entrada (onde estão as imagens e labels originais)
IMG_INPUT_DIR = "temp"
LABEL_INPUT_DIR = "labels"

# Arquivo CSV que mapeia cada imagem para seu split (train/test/val)
CSV_FILE = "MITOS_WSI_CCMCT/outputs/CCMCT_split.csv"

# Pasta raiz de saída (serão criadas subpastas: saida/images/train, saida/images/test, etc)
OUTPUT_BASE_DIR = "MITOS_WSI_CCMCT/outputs"

# Parâmetros do tiling (veja tile_label_creator.py para detalhes)
TILE_SIZE = 320
OVERLAP = 64
MIN_VISIBILITY = 0.5
KEEP_EMPTY_RATIO = 0.0  # não guardar tiles vazios


# ==============================================================================
# FUNÇÕES
# ==============================================================================

def load_csv_mapping(csv_path):
    """
    Lê o CSV usando pandas e retorna um dicionário que mapeia número da imagem → split.
    
    Esperado: CSV com colunas "filename" e "split"
    Exemplo: filename=001.tiff → split=train
    
    Retorna: dict { imagem_number: split }
    """
    try:
        # Lê o CSV com ';' como separador
        df = pd.read_csv(csv_path)
        
        # Valida colunas necessárias
        required_cols = {'filename', 'split'}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise KeyError(f"Colunas faltando no CSV: {missing}. Esperado: 'filename' e 'split'")
        
        # Cria mapeamento: número da imagem → split
        mapping = {}
        for _, row in df.iterrows():
            try:
                # Extrai o número do filename (ex: "001.tiff" → 1)
                img_num = row['filename'].split('.')[0]
                print(f'filename, {img_num}')
                split = row['split'].lower().strip()
                mapping[img_num] = split
            except (ValueError, TypeError):
                # Ignora linhas onde filename não é conversível para int
                continue
        
        return mapping
        
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Erro ao ler CSV: {e}")


def get_split_for_image_midogpp(image_path, csv_mapping):
    """
    Extrai o número da imagem do caminho e retorna o split correspondente.
    
    Exemplo: "001.tiff" → número 1 → "train"
    """
    stem = image_path.stem  # "001" de "001.tiff"
    
    try:
        image_num = int(stem)
    except ValueError:
        return None  # arquivo não segue padrão numérico
    
    return csv_mapping.get(image_num)


def get_split_for_image(image_path, csv_mapping):
    """
    Extrai o número da imagem do caminho e retorna o split correspondente.
    
    Exemplo: "001.tiff" → número 1 → "train"
    """
    
    image_name = str(image_path).split('/')[2]
    print(image_name)
    
    try:
        csv_mapping.get(image_name)
    except ValueError:
        return None  # arquivo não segue padrão numérico
    
    return csv_mapping.get(image_name)


def ensure_output_dirs(base_dir, splits):
    """Cria a estrutura de pastas de saída se não existirem."""
    for split in splits:
        img_dir = Path(base_dir) / "images" / split
        label_dir = Path(base_dir) / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)


def process_all_images(img_dir, label_dir, csv_file, output_base_dir, 
                       tile_size=640, overlap=128, min_visibility=0.5, keep_empty_ratio=0.0):
    """
    Processa todas as imagens da pasta de entrada.
    
    Para cada imagem:
      1. Encontra o label correspondente
      2. Consulta o CSV para descobrir o split
      3. Cria a pasta de saída para esse split
      4. Chama tile_image_with_labels() com os caminhos corretos
      5. Imprime resumo do processamento
    """
    
    img_dir = Path(img_dir)
    label_dir = Path(label_dir)
    output_base_dir = Path(output_base_dir)
    
    # =========================================================================
    # Etapa 1: Carregar o mapeamento do CSV
    # =========================================================================
    print(f"📂 Carregando mapeamento de splits de '{csv_file}'...")
    csv_mapping = load_csv_mapping(csv_file)
    unique_splits = set(csv_mapping.values())
    print(f"   ✓ Encontrados {len(unique_splits)} splits: {', '.join(sorted(unique_splits))}\n")
    
    # Cria a estrutura de pastas de saída
    ensure_output_dirs(output_base_dir, unique_splits)
    
    # =========================================================================
    # Etapa 2: Listar todas as imagens de entrada
    # =========================================================================
    # Procura por imagens com extensões comuns
    image_extensions = {'.tiff', '.tif', '.jpg', '.jpeg', '.png'}
    image_files = sorted([
        f for f in img_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ])
    
    if not image_files:
        print(f"Nenhuma imagem encontrada em '{img_dir}'")
        return
    
    print(f"Encontradas {len(image_files)} imagens em '{img_dir}'\n")
    
    # =========================================================================
    # Etapa 3: Processar cada imagem
    # =========================================================================
    processed = 0
    skipped = 0
    errors = 0
    
    for img_path in image_files:
        # Descobre o split desta imagem
        split = get_split_for_image(img_path, csv_mapping)
        
        if split is None:
            print(f"[!]  '{img_path.name}' — nome não segue padrão numérico, pulando")
            skipped += 1
            continue
        
        # Encontra o label correspondente
        label_path = label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            print(f"[!]  '{img_path.name}' — label não encontrado ({label_path.name}), pulando")
            skipped += 1
            continue
        
        # Pastas de saída para este split
        out_img_dir = output_base_dir / "images" / split
        out_label_dir = output_base_dir / "labels" / split
        
        # =====================================================================
        # Processa a imagem
        # =====================================================================
        try:
            print(f"Processando '{img_path.name}' (split: {split})...")
            
            saved = tile_image_with_labels(
                str(img_path),
                str(label_path),
                str(out_img_dir),
                str(out_label_dir),
                tile=tile_size,
                overlap=overlap,
                min_visibility=min_visibility,
                keep_empty_ratio=keep_empty_ratio
            )
            
            print(f"[ok] {saved} tiles salvos em '{split}'\n")
            processed += 1
            
        except Exception as e:
            print(f"[x] Erro ao processar: {e}\n")
            errors += 1
    
    # =========================================================================
    # Resumo final
    # =========================================================================
    print("=" * 70)
    print("RESUMO DO PROCESSAMENTO")
    print("=" * 70)
    print(f"[ok] Processadas com sucesso: {processed}")
    print(f"[!]  Puladas: {skipped}")
    print(f"[x] Erros: {errors}")
    print(f"Saída: {output_base_dir}")
    print("=" * 70)


# ==============================================================================
# PONTO DE ENTRADA
# ==============================================================================

if __name__ == "__main__":

    folders = os.listdir(OUTPUT_BASE_DIR)
    folders = [f for f in folders if Path(f'{OUTPUT_BASE_DIR}/{f}').is_dir()]
        
    for folder in folders:        
        print("\n" + "=" * 70)
        print("BATCH TILER - Processa todas as imagens em tiles")
        print("=" * 70 + "\n")

        IMG_INPUT_DIR = f'{OUTPUT_BASE_DIR}/{folder}/images'
        LABEL_INPUT_DIR = f'{OUTPUT_BASE_DIR}/{folder}/yolo_labels'
        OUTPUT_DIR = f'{OUTPUT_BASE_DIR}/{folder}/patches'

        if not Path(OUTPUT_DIR).exists(): os.makedirs(OUTPUT_DIR)
        
        # Valida pastas de entrada
        if not Path(IMG_INPUT_DIR).exists():
            print(f"[x] Pasta de entrada '{IMG_INPUT_DIR}' não existe")
            exit(1)
        
        if not Path(LABEL_INPUT_DIR).exists():
            print(f"[x] Pasta de labels '{LABEL_INPUT_DIR}' não existe")
            exit(1)
        
        if not Path(CSV_FILE).exists():
            print(f"[x] Arquivo CSV '{CSV_FILE}' não existe")
            exit(1)
        
        # Executa o processamento
        process_all_images(
            IMG_INPUT_DIR,
            LABEL_INPUT_DIR,
            CSV_FILE,
            OUTPUT_DIR,
            tile_size=TILE_SIZE,
            overlap=OVERLAP,
            min_visibility=MIN_VISIBILITY,
            keep_empty_ratio=KEEP_EMPTY_RATIO
        )
        
        print(f"\n[ok] Processamento concluído! - {OUTPUT_BASE_DIR}/{folder}\n")
