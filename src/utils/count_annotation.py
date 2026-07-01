#!/usr/bin/env python3
"""
Conta o número TOTAL de anotações (bounding boxes) em arquivos de label YOLO,
dentro de todas as pastas de treino do dataset MITOS_WSI_CMC.

Diferença importante:
  - "quantidade de arquivos de label" = número de arquivos .txt
  - "quantidade de anotações"          = número de LINHAS não vazias somadas
                                          em todos esses arquivos
    (cada linha de um label YOLO representa uma anotação/bounding box:
     classe x_center y_center width height)

Estrutura esperada (uma pasta por execução/hash):
  yolo26-mitoses-detection/MITOS_WSI_CMC/outputs/<hash>/patches/labels/train/*.txt

Exemplo de uso:
  python count_annotations.py
  python count_annotations.py --root yolo26-mitoses-detection/MITOS_WSI_CMC/outputs --verbose
  python count_annotations.py --root /caminho/absoluto/outputs --pattern "*/patches/labels/train"
"""

import argparse
from pathlib import Path


def count_annotations_in_file(filepath: Path) -> int:
    """Conta linhas não vazias (= anotações) em um arquivo de label YOLO."""
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def find_train_label_dirs(root: Path, pattern: str) -> list[Path]:
    """Encontra todas as pastas labels/train sob root, seguindo o padrão glob."""
    return sorted(p for p in root.glob(pattern) if p.is_dir())


def main():
    parser = argparse.ArgumentParser(
        description="Conta anotações totais (bounding boxes) em labels YOLO de treino do MIDOGpp."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="MIDOGpp/outputs",
        help="Pasta base 'outputs', contendo uma subpasta por experimento/hash "
             "(padrão: %(default)s)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="labels/train",
        help="Padrão glob (relativo ao --root) para localizar as pastas labels/train "
             "(padrão: %(default)s)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mostra a contagem detalhada por subpasta/hash.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Pasta não encontrada: {root}")

    label_dirs = find_train_label_dirs(root, args.pattern)
    if not label_dirs:
        raise SystemExit(
            f"Nenhuma pasta encontrada com o padrão '{args.pattern}' em '{root}'."
        )

    total_annotations = 0
    total_files = 0
    total_empty_files = 0
    per_dir_summary = []

    for label_dir in label_dirs:
        txt_files = sorted(label_dir.glob("*.txt"))
        dir_annotations = 0
        dir_empty = 0

        for txt_file in txt_files:
            n = count_annotations_in_file(txt_file)
            dir_annotations += n
            if n == 0:
                dir_empty += 1

        total_annotations += dir_annotations
        total_files += len(txt_files)
        total_empty_files += dir_empty
        per_dir_summary.append((label_dir, len(txt_files), dir_annotations, dir_empty))

    print("=" * 72)
    print("CONTAGEM DE ANOTAÇÕES — MIDOGpp (train)")
    print("=" * 72)

    if args.verbose:
        for label_dir, n_files, n_ann, n_empty in per_dir_summary:
            try:
                rel = label_dir.relative_to(root)
            except ValueError:
                rel = label_dir
            print(f"\n{rel}")
            print(f"  arquivos de label (.txt) : {n_files}")
            print(f"  anotações                : {n_ann}")
            print(f"  arquivos vazios (0 anot.) : {n_empty}")
        print("\n" + "-" * 72)

    print(f"Subpastas 'labels/train' encontradas    : {len(label_dirs)}")
    print(f"Total de arquivos de label (.txt)       : {total_files}")
    print(f"Total de arquivos vazios (0 anotações)  : {total_empty_files}")
    print(f"TOTAL DE ANOTAÇÕES (bounding boxes)     : {total_annotations}")
    if total_files:
        print(f"Média de anotações por arquivo          : {total_annotations / total_files:.2f}")
    print("=" * 72)


if __name__ == "__main__":
    main()