import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

try:
    import torchstain
except ImportError:
    print("Erro: torchstain não está instalado!")
    print("Instale com: pip install torchstain")
    sys.exit(1)

IMAGE_EXTENSIONS = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}

T = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 255)
])


def build_normalizer(target_img):
    """Fita o normalizer uma única vez no target."""
    normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
    normalizer.fit(T(target_img))
    return normalizer


def macenko_normalization_torch(image_path, normalizer, output_path):
    raw = cv2.imread(str(image_path))
    if raw is None:
        return False

    to_transform = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    norm, H, E = normalizer.normalize(I=T(to_transform), stains=True)

    img_np = norm.numpy().astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img_bgr)
    return True


def collect_images(input_dir):
    """Coleta recursiva de imagens."""
    return [
        p for p in Path(input_dir).rglob('*')
        if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Normalização de stain Macenko via torchstain"
    )
    parser.add_argument('--input',  '-i', required=True,  help='Pasta de entrada')
    parser.add_argument('--output', '-o', default=None,    help='Pasta de saída')
    parser.add_argument('--reference', '-r', required=True, help='Imagem de referência')
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir.parent / f"{input_dir.name}_macenko"

    print("=" * 60)
    print("   Normalização de Stain — Método Macenko (torchstain)")
    print("=" * 60)

    target = cv2.cvtColor(cv2.imread(args.reference), cv2.COLOR_BGR2RGB)
    normalizer = build_normalizer(target)

    images = collect_images(input_dir)
    processed = 0
    failed = 0

    for img_path in tqdm(images, desc="Processando"):
        relative = img_path.relative_to(input_dir)
        out_path = output_dir / relative
        if macenko_normalization_torch(img_path, normalizer, out_path):
            processed += 1
        else:
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Processamento concluído!")
    print(f"  ✓ Sucesso: {processed}")
    print(f"  ✗ Falha:   {failed}")
    print(f"  Total:     {len(images)}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()