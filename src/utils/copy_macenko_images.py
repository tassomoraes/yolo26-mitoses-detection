#!/usr/bin/env python3
"""Copiar imagens Macenko para o repositório MIDOG_CMC_CCMCT."""

import argparse
import shutil
from pathlib import Path
import os


IMAGE_EXTENSIONS = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp', '.gif'}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def copy_images(src_dir: Path, dest_dir: Path, prefix: str = ''):
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = 0

    for path in src_dir.iterdir():
        if is_image_file(path):
            dest_path = dest_dir / (f"{prefix}_{path.name}" if prefix else path.name)
            if dest_path.exists():
                suffix = 1
                stem = path.stem
                while True:
                    candidate = dest_dir / f"{prefix}_{stem}_{suffix}{path.suffix}" if prefix else dest_dir / f"{stem}_{suffix}{path.suffix}"
                    if not candidate.exists():
                        dest_path = candidate
                        break
                    suffix += 1
            shutil.copy2(path, dest_path)
            copied += 1
    return copied


def main():
    parser = argparse.ArgumentParser(
        description='Copiar imagens *_macenko de MITOS_WSI_CMC/outputs para MIDOG_CMC_CCMCT/macenko_images'
    )
    parser.add_argument('--outputs-root', default='MITOS_WSI_CMC/outputs', help='Raiz de saída de MITOS_WSI_CCMCT')
    parser.add_argument('--dest-root', default='MIDOG_CMC_CMC/macenko_images', help='Raiz de destino para imagens Macenko')
    parser.add_argument('--keep-folder-prefix', action='store_true', help='Prefixar nomes de arquivos com o ID da pasta de origem para evitar colisões')
    args = parser.parse_args()

    root = args.outputs_root

    folders = os.listdir(root)
    folders = [f for f in folders if Path(f'{root}/{f}').is_dir()]
    
    count = 0

    for folder in folders:
        path = f'{root}/{folder}/patches/images'
        dirs = os.listdir(path)

        macenko_dir = [f for f in dirs if '_macenko' in f][0]
        split_dir = [f.split('_')[0] for f in dirs if '_macenko' in f][0]

        origin_path = f'{path}/{macenko_dir}/'
        dst_path = f'MIDOG_CMC_CCMCT/images/{split_dir}'

        count_file = 0

        for file in os.listdir(origin_path):
            
            # print(f'Copy img: {origin_path}{file}', f'{dst_path}/{file}')
            shutil.copy2(f'{origin_path}{file}', f'{dst_path}/{file}')

            original_label_path = path
            original_label_path = f'{original_label_path.replace('images','labels')}/{split_dir}'
            dst_label_path = dst_path
            dst_label_path = dst_label_path.replace('images','labels')

            label_file = file.replace('jpg', 'txt')
            # print(f'Copy label: {original_label_path}/{label_file}', f'{dst_label_path}/{label_file}')
            shutil.copy2(f'{original_label_path}/{label_file}', f'{dst_label_path}/{label_file}')
            count_file += 1

        print(f'{count_file} imagens e labels copiadas!')

        count = count + count_file

    print(f'Total de {count} imagens e labels copiadas!')

    '''
    repo_root = Path(__file__).resolve().parent.parent
    outputs_root = (repo_root / args.outputs_root).resolve()
    dest_root = (repo_root / args.dest_root).resolve()

    if not outputs_root.exists():
        raise FileNotFoundError(f'Pasta de outputs não encontrada: {outputs_root}')

    total_copied = 0
    for slide_dir in sorted(outputs_root.iterdir()):
        if not slide_dir.is_dir():
            continue

        patches_images = slide_dir / 'patches' / 'images'
        if not patches_images.exists():
            continue

        for child in sorted(patches_images.iterdir()):
            if not child.is_dir() or not child.name.endswith('_macenko'):
                continue

            split = child.name[:-len('_macenko')]
            if not split:
                continue

            dest_dir = dest_root / split
            prefix = slide_dir.name if args.keep_folder_prefix else ''
            print(f'cpoy img: {child}, {dest_dir}, {prefix}')
            #copied = copy_images(child, dest_dir, prefix)
            #total_copied += copied
            #print(f'Copiadas {copied} imagens de {child} para {dest_dir}')
    
    print(f'Total de imagens copiadas: {total_copied}')
    '''


if __name__ == '__main__':
    main()
