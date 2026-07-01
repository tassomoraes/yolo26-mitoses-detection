#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path
import os


def main():
    parser = argparse.ArgumentParser(description='Aplicar normalização Macenko em pastas de imagens de outputs')
    parser.add_argument('--reference', '-r', default='data/reference_img.jpg', help='Caminho da imagem de referência')
    parser.add_argument('--root', default='MITOS_WSI_CCMCT/outputs', help='Diretório raiz com as pastas de outputs')
    parser.add_argument('--python', default=sys.executable, help='Executável do Python')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    reference = Path(args.reference).resolve() if not Path(args.reference).is_absolute() else Path(args.reference)
    root = Path(args.root).resolve() if not Path(args.root).is_absolute() else Path(args.root)
    script = Path('src/apply_macenko_stain.py').resolve()

    if not reference.exists():
        raise FileNotFoundError(f'Imagem de referência não encontrada: {reference}')
    if not script.exists():
        raise FileNotFoundError(f'Script não encontrado: {script}')
    if not root.exists():
        raise FileNotFoundError(f'Pasta raiz não encontrada: {root}')

    
    folders = os.listdir(root)
    folders = [f for f in folders if Path(f'{root}/{f}').is_dir()]
        
    image_dirs = []

    for folder in folders: 
        image_dirs.append(root / folder / 'patches' / 'images' / 'train')
        image_dirs.append(root / folder / 'patches' / 'images' / 'val')


    for image_dir in image_dirs:
        if not image_dir.exists():
            print(f'Ignorando pasta inexistente: {image_dir}')
            continue

        #print(f'[INFO] - name: {image_dir}')
        
        output_dir = image_dir.parent / f"{image_dir.name}_macenko"
        cmd = [
            args.python,
            str(script),
            '--input', str(image_dir),
            '--output', str(output_dir),
            '--reference', str(reference),
        ]
        print('Executando:', ' '.join(cmd))
        subprocess.run(cmd, check=False)
        


if __name__ == '__main__':
    main()
