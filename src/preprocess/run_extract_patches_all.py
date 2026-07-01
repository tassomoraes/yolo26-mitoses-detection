#!/usr/bin/env python3
"""
Batch runner for extract_patches.py

For each whole-slide image in ../WSI, creates an output folder
in ../outputs/<image_basename> and invokes extract_patches.py with it.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def find_wsi_files(wsi_dir):
    exts = {'.svs', '.tif', '.tiff', '.ndpi', '.mrxs', '.vms'}
    for p in Path(wsi_dir).iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def main():
    this_dir = Path(__file__).resolve().parent
    default_wsi = this_dir.parent / 'WSI'
    default_outputs = this_dir.parent / 'outputs'
    default_extract = this_dir / 'extract_patches.py'

    parser = argparse.ArgumentParser(description='Run extract_patches.py on all WSI files in a folder')
    parser.add_argument('--wsi-dir', type=Path, default=default_wsi,
                        help='Directory containing WSI files (default: ../WSI)')
    parser.add_argument('--outputs-dir', type=Path, default=default_outputs,
                        help='Parent outputs directory (default: ../outputs)')
    parser.add_argument('--extract-script', type=Path, default=default_extract,
                        help='Path to extract_patches.py (default: src/extract_patches.py)')
    parser.add_argument('--db', type=Path, required=True,
                        help='Path to the sqlite database to pass as --db')
    parser.add_argument('--pw', '--patch-width', dest='pw', type=int, required=True,
                        help='Patch width in pixels (passed as -pw)')
    parser.add_argument('--ph', '--patch-height', dest='ph', type=int, required=True,
                        help='Patch height in pixels (passed as -ph)')
    parser.add_argument('--python', type=Path, default=Path(sys.executable),
                        help='Python executable to run (default: current interpreter)')
    args = parser.parse_args()

    wsi_dir = args.wsi_dir
    outputs_dir = args.outputs_dir
    extract_script = args.extract_script

    outputs_dir.mkdir(parents=True, exist_ok=True)

    files = list(find_wsi_files(wsi_dir))
    if not files:
        print(f'No WSI files found in {wsi_dir}')
        return

    for p in files:
        image_id = p.stem
        out_dir = outputs_dir / image_id
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [str(args.python), str(extract_script),
               '-pw', str(args.pw), '-ph', str(args.ph),
               '--svs', str(p), '--db', str(args.db),
               '--output', str(out_dir)]

        print('\nRunning:', ' '.join(cmd))
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            print(f'Warning: extract_patches.py failed for {p} (exit {proc.returncode})')


if __name__ == '__main__':
    main()
