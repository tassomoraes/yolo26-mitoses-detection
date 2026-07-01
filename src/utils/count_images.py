#!/usr/bin/env python3
"""
Conta a quantidade de IMAGENS em cada pasta de treino do dataset MITOS_WSI_CMC,
e o total geral — seguindo a mesma lógica do count_annotations.py.

Estrutura esperada (uma pasta por execução/hash):
  yolo26-mitoses-detection/MITOS_WSI_CMC/outputs/<hash>/patches/images/train/*

Apenas arquivos com extensão de imagem são contados (ver --extensions);
arquivos auxiliares (.txt, .json, .cache, etc.) dentro da mesma pasta são ignorados.

Exemplo de uso:
  python count_images.py
  python count_images.py --root yolo26-mitoses-detection/MITOS_WSI_CMC/outputs --verbose
  python count_images.py --extensions .png .tif .tiff
"""

import argparse
from pathlib import Path

DEFAULT_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]


def find_train_image_dirs(root: Path, pattern: str) -> list[Path]:
    """Encontra todas as pastas images/train sob root, seguindo o padrão glob."""
    return sorted(p for p in root.glob(pattern) if p.is_dir())


def count_images_in_dir(image_dir: Path, extensions: set[str]) -> int:
    """Conta arquivos cuja extensão (case-insensitive) esteja em `extensions`."""
    return sum(
        1
        for f in image_dir.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    )


def main():
    parser = argparse.ArgumentParser(
        description="Conta a quantidade de imagens em cada pasta patches/images/train do MIDOGpp."
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
        default="images/train",
        help="Padrão glob (relativo ao --root) para localizar as pastas images/train "
             "(padrão: %(default)s)",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=DEFAULT_EXTENSIONS,
        help=f"Extensões de imagem a considerar (padrão: {' '.join(DEFAULT_EXTENSIONS)})",
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

    extensions = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in args.extensions}

    image_dirs = find_train_image_dirs(root, args.pattern)
    if not image_dirs:
        raise SystemExit(
            f"Nenhuma pasta encontrada com o padrão '{args.pattern}' em '{root}'."
        )

    total_images = 0
    per_dir_summary = []

    for image_dir in image_dirs:
        n_images = count_images_in_dir(image_dir, extensions)
        total_images += n_images
        per_dir_summary.append((image_dir, n_images))

    print("=" * 72)
    print("CONTAGEM DE IMAGENS — MIDOGpp (train)")
    print("=" * 72)

    if args.verbose:
        for image_dir, n_images in per_dir_summary:
            try:
                rel = image_dir.relative_to(root)
            except ValueError:
                rel = image_dir
            print(f"\n{rel}")
            print(f"  imagens : {n_images}")
        print("\n" + "-" * 72)

    print(f"Subpastas 'images/train' encontradas : {len(image_dirs)}")
    print(f"TOTAL DE IMAGENS                     : {total_images}")
    if image_dirs:
        print(f"Média de imagens por subpasta        : {total_images / len(image_dirs):.2f}")
    print("=" * 72)


if __name__ == "__main__":
    main()