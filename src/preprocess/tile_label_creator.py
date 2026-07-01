"""
Tiling de imagens grandes (lâminas de patologia) para treino YOLO.

Pipeline: imagem gigante + anotações YOLO  ->  vários tiles 640x640 + anotações locais.

Etapas, na ordem em que rodam:
  1. load_yolo_labels    -> lê o .txt YOLO e converte para caixas em PIXELS absolutos
  2. compute_tile_grid   -> calcula as posições (canto superior-esquerdo) de cada tile
  3. remap_boxes_to_tile -> para um tile específico, decide quais caixas entram e reajusta coordenadas
  4. should_save_tile    -> aplica a política de balanceamento (tiles vazios x com objeto)
  5. save_tile           -> grava a imagem recortada + o .txt de anotações
  6. tile_image_with_labels -> orquestra tudo, varrendo a grade
"""

import os
import random
from pathlib import Path
import cv2


# ---------------------------------------------------------------------------
# Estrutura de dados
# ---------------------------------------------------------------------------
# Uma "caixa em pixels" é uma tupla: (classe, x1, y1, x2, y2)
#   - classe: int, o id da classe (ex.: 0 = mitose)
#   - x1, y1: canto superior-esquerdo, em pixels absolutos da imagem original
#   - x2, y2: canto inferior-direito, em pixels absolutos
# Trabalhar em pixels absolutos (e não no formato YOLO normalizado) durante o
# processamento evita erros de normalização: só normalizamos de volta no fim.


# ---------------------------------------------------------------------------
# Etapa 1 — Carregar anotações
# ---------------------------------------------------------------------------
def load_yolo_labels(label_path, img_w, img_h):
    """
    Lê um arquivo de anotações no formato YOLO e converte para caixas em pixels.

    Formato YOLO de entrada (uma linha por objeto, valores normalizados 0-1):
        classe  centro_x  centro_y  largura  altura

    Retorna: lista de tuplas (classe, x1, y1, x2, y2) em pixels absolutos.
             Lista vazia se o arquivo não existir (tile/imagem sem objetos).
    """
    boxes = []
    if not os.path.exists(label_path):
        return boxes

    for line in open(label_path):
        line = line.strip()
        if not line:
            continue
        cls, cx, cy, bw, bh = map(float, line.split())

        # Converte centro+tamanho (normalizado) -> cantos (em pixels)
        x1 = (cx - bw / 2) * img_w
        y1 = (cy - bh / 2) * img_h
        x2 = (cx + bw / 2) * img_w
        y2 = (cy + bh / 2) * img_h

        boxes.append((int(cls), x1, y1, x2, y2))

    return boxes


# ---------------------------------------------------------------------------
# Etapa 2 — Calcular a grade de tiles
# ---------------------------------------------------------------------------
def compute_tile_grid(img_w, img_h, tile, overlap):
    """
    Calcula as posições (x, y) do canto superior-esquerdo de cada tile.

    'stride' é o passo entre tiles consecutivos. Com overlap > 0, tiles
    vizinhos se sobrepõem — isso garante que objetos pequenos perto de uma
    borda apareçam INTEIROS em pelo menos um tile.

    O ajuste final de cada eixo "ancora" um tile no fim da imagem (img_w - tile).
    Sem isso, a faixa da direita/inferior que não é múltipla do stride
    simplesmente não seria coberta.

    Retorna: lista de tuplas (tx, ty), os cantos de cada tile.
    """
    stride = tile - overlap

    def axis_positions(total, tile, stride):
        # Posições regulares espaçadas pelo stride.
        positions = list(range(0, max(total - tile, 0) + 1, stride)) or [0]
        # Garante cobertura da borda final, se a imagem for maior que um tile.
        if total > tile and positions[-1] != total - tile:
            positions.append(total - tile)
        return positions

    xs = axis_positions(img_w, tile, stride)
    ys = axis_positions(img_h, tile, stride)

    # Produto cartesiano: todas as combinações (coluna, linha) da grade.
    return [(tx, ty) for ty in ys for tx in xs]


# ---------------------------------------------------------------------------
# Etapa 3 — Reposicionar caixas para um tile específico
# ---------------------------------------------------------------------------
def remap_boxes_to_tile(boxes, tx, ty, tile, min_visibility):
    """
    Dado um tile na posição (tx, ty), descobre quais caixas caem dentro dele
    e converte suas coordenadas para o sistema local do tile (formato YOLO).

    Para cada caixa original:
      a) calcula a INTERSEÇÃO entre a caixa e o retângulo do tile;
      b) se não há interseção -> descarta;
      c) se a caixa está MUITO cortada (fração visível < min_visibility da
         área ORIGINAL) -> descarta, pois vira um rótulo ruim;
      d) caso contrário -> recorta (clip) a caixa aos limites do tile e
         converte para coordenadas locais normalizadas.

    Nota sobre (c): a fração é medida contra a área original da caixa, não a
    visível. Uma mitose grande cortada pela metade ainda é reconhecível e vale
    manter; uma cortada em 10% não. Como o overlap costuma ser maior que uma
    mitose, na prática esse descarte quase nunca atinge mitoses reais — ele
    funciona como rede de segurança.

    Retorna: lista de tuplas (classe, cx, cy, w, h) normalizadas 0-1 pelo tile.
    """
    tile_boxes = []

    for cls, x1, y1, x2, y2 in boxes:
        # (a) Interseção entre a caixa [x1,y1,x2,y2] e o tile [tx,ty,tx+tile,ty+tile]
        ix1 = max(x1, tx)
        iy1 = max(y1, ty)
        ix2 = min(x2, tx + tile)
        iy2 = min(y2, ty + tile)

        # (b) Sem sobreposição: a caixa não pertence a este tile.
        if ix2 <= ix1 or iy2 <= iy1:
            continue

        # (c) Caixa muito cortada pela borda do tile -> descarta.
        original_area = (x2 - x1) * (y2 - y1)
        visible_area = (ix2 - ix1) * (iy2 - iy1)
        if original_area <= 0 or visible_area / original_area < min_visibility:
            continue

        # (d) Coordenadas locais ao tile (origem no canto do tile), já recortadas.
        local_x1 = ix1 - tx
        local_y1 = iy1 - ty
        local_x2 = ix2 - tx
        local_y2 = iy2 - ty

        # Converte cantos -> centro+tamanho, normalizado pelo tamanho do tile.
        cx = (local_x1 + local_x2) / 2 / tile
        cy = (local_y1 + local_y2) / 2 / tile
        bw = (local_x2 - local_x1) / tile
        bh = (local_y2 - local_y1) / tile

        tile_boxes.append((cls, cx, cy, bw, bh))

    return tile_boxes


# ---------------------------------------------------------------------------
# Etapa 4 — Política de balanceamento
# ---------------------------------------------------------------------------
def should_save_tile(tile_boxes, keep_empty_ratio):
    """
    Decide se um tile deve ser salvo.

    - Tile COM objeto: sempre salvo (todo positivo é valioso e escasso).
    - Tile VAZIO: salvo só com probabilidade 'keep_empty_ratio'. A rede precisa
      de alguns negativos (tecido sem mitose), mas a maioria esmagadora dos
      tiles de uma lâmina é vazia — guardar todos desbalancearia o dataset.

    keep_empty_ratio = 0.0  -> nenhum tile vazio
    keep_empty_ratio = 1.0  -> todos os tiles vazios
    keep_empty_ratio = 0.1  -> ~10% dos tiles vazios
    """
    if tile_boxes:
        return True
    return random.random() < keep_empty_ratio


# ---------------------------------------------------------------------------
# Etapa 5 — Gravar o tile em disco
# ---------------------------------------------------------------------------
def save_tile(img, tx, ty, tile, tile_boxes, stem, out_img_dir, out_label_dir):
    """
    Recorta o tile da imagem original e grava o par (imagem .jpg + rótulo .txt).

    O nome do arquivo embute a posição (stem_tx_ty), o que facilita rastrear
    de qual região da lâmina cada tile veio — útil na inspeção e na remontagem
    durante a inferência.
    """
    crop = img[ty:ty + tile, tx:tx + tile]
    name = f"{stem}_{tx}_{ty}"

    cv2.imwrite(os.path.join(out_img_dir, f"{name}.jpg"), crop)

    with open(os.path.join(out_label_dir, f"{name}.txt"), "w") as f:
        for cls, cx, cy, bw, bh in tile_boxes:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


# ---------------------------------------------------------------------------
# Etapa 6 — Orquestrador
# ---------------------------------------------------------------------------
def tile_image_with_labels(
    img_path, label_path, out_img_dir, out_label_dir,
    tile=640, overlap=128, min_visibility=0.5, keep_empty_ratio=0.0
):
    """
    Fatia uma imagem grande + suas anotações YOLO numa coleção de tiles.

    Parâmetros:
      img_path, label_path   : caminhos da imagem e do .txt YOLO de entrada
      out_img_dir, out_label_dir : pastas de saída (devem existir)
      tile                   : lado do tile em pixels (ex.: 640)
      overlap                : sobreposição entre tiles vizinhos (ex.: 128 = 20%)
      min_visibility         : fração mínima da caixa que deve estar no tile
      keep_empty_ratio       : probabilidade de manter um tile sem objetos

    Apenas costura as 5 etapas acima — toda a lógica vive nelas.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Não consegui ler a imagem: {img_path}")
    img_h, img_w = img.shape[:2]
    stem = Path(img_path).stem

    # Etapa 1: anotações originais -> caixas em pixels
    boxes = load_yolo_labels(label_path, img_w, img_h)

    # Etapa 2: posições de todos os tiles
    grid = compute_tile_grid(img_w, img_h, tile, overlap)

    saved = 0
    for tx, ty in grid:
        # Etapa 3: quais caixas entram neste tile, em coordenadas locais
        tile_boxes = remap_boxes_to_tile(boxes, tx, ty, tile, min_visibility)

        # Etapa 4: decide se vale salvar (balanceamento vazio x positivo)
        if not should_save_tile(tile_boxes, keep_empty_ratio):
            continue

        # Etapa 5: grava imagem + rótulo
        save_tile(img, tx, ty, tile, tile_boxes, stem,
                  out_img_dir, out_label_dir)
        saved += 1

    return saved  # nº de tiles efetivamente gravados — útil para logs/sanidade

# ---------------------------------------------------------------------------
# Função auxiliar para verificar visualmente as caixas em um tile
# ---------------------------------------------------------------------------
def draw_yolo_check(image_path, label_path, save=True):
    # Carrega a imagem
    img = cv2.imread(image_path)
    h, w, _ = img.shape

    # Lê o arquivo de label
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        # Formato YOLO: class x_center y_center width height
        parts = line.split()
        cls = parts[0]
        x_c, y_c, bw, bh = map(float, parts[1:])

        # Inverte a normalização para pixels
        # x_min = (x_center - width/2) * largura_da_imagem
        xmin = int((x_c - bw / 2) * w)
        ymin = int((y_c - bh / 2) * h)
        xmax = int((x_c + bw / 2) * w)
        ymax = int((y_c + bh / 2) * h)

        # Desenha o retângulo (Cor Verde, espessura 2, se classe for 0) ou (Cor Vermelha, espessura 2, se classe for 1)
        if cls == "0":
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(img, f"Classe: {cls}", (xmin, ymin - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            cv2.putText(img, f"Classe: {cls}", (xmin, ymin - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Mostra o resultado
    cv2.imshow("Verificacao YOLO", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Salva a imagem com as caixas desenhadas
    if save:
        cv2.imwrite(f'checkbox_imgs/image_{image_path.split("/")[-1].split(".")[0]}_yolo_check.tiff', img)

# Exemplo de uso:
# draw_yolo_check('out/images/001_512_512.jpg', 'out/labels/001_512_512.txt') 


# Exemplo de uso do orquestrador:
# saved_tiles = tile_image_with_labels(
#    img_path='input/001.tiff',
#    label_path='input/001.txt',
#    out_img_dir='out/images',
#    out_label_dir='out/labels',
#    tile=640,
#    overlap=128,
#    min_visibility=0.5,
#    keep_empty_ratio=0.1
#)
#print(f"Tiles salvos: {saved_tiles}")