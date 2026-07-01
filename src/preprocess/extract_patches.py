import argparse
import os
import sqlite3
import sys

import openslide

from coordinates import get_patch_coordinates
from filter_petch import filter_patches_parallel
from relative_annotation import get_relative_annotations_with_class
from yolo_label_converter import converter_para_yolo_paralelo


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract WSI patches and their annotations.",
        add_help=False,
    )
    parser.add_argument('-pw', '--patch-width', type=int, required=True,
                        help="Patch width in pixels")
    parser.add_argument('-ph', '--patch-height', type=int, required=True,
                        help="Patch height in pixels")
    parser.add_argument('--svs', type=str, required=True,
                        help="Path to the .svs file")
    parser.add_argument('--db', type=str, required=True,
                        help="Path to the .sqlite database")
    parser.add_argument('--output', type=str, default="./patches",
                        help="Output directory (default: ./patches)")
    parser.add_argument('--help', action='help', help="Show this help message and exit")
    return parser.parse_args()


def load_slide_annotations(db_path, basename):
    """
    Returns (slide_uid, annotation_points, class_map).
    Exits the process if the slide is not found.
    """
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT uid FROM Slides WHERE filename = ?", (basename,))
        row = cur.fetchone()
        if row is None:
            print(f"Error: slide '{basename}' not found in database {db_path}",
                  file=sys.stderr)
            sys.exit(1)
        slide_uid = row[0]

        cur.execute(
            """
            SELECT ac.coordinateX, ac.coordinateY, a.agreedClass
            FROM Annotations a
            JOIN Annotations_coordinates ac ON a.uid = ac.annoId
            WHERE a.slide = ?
              AND a.deleted = 0
              AND a.agreedClass IN (1, 2)
              AND ac.orderIdx = 1
            """,
            (slide_uid,),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    annotation_points = [(row[1], row[0]) for row in rows]
    class_map = {
        (row[1], row[0]): ("mitose" if row[2] == 2 else "nao_mitose")
        for row in rows
    }
    return slide_uid, annotation_points, class_map


def main():
    args = parse_args()

    svs_path = args.svs
    db_path = args.db
    output_dir = args.output
    patch_w = args.patch_width
    patch_h = args.patch_height

    basename = os.path.basename(svs_path)
    image_id = os.path.splitext(basename)[0]

    slide = openslide.OpenSlide(svs_path)
    image_width, image_height = slide.dimensions

    _, annotation_points, class_map = load_slide_annotations(db_path, basename)

    if not annotation_points:
        print(f"No valid annotations (mitose/nao_mitose) for slide '{basename}'. Nothing to extract.")
        slide.close()
        return

    all_patches = get_patch_coordinates((image_height, image_width), (patch_h, patch_w))
    valid_patches = filter_patches_parallel(all_patches, annotation_points)

    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    yolo_labels_dir = os.path.join(output_dir, "yolo_labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    annotations_with_class = [
        (y, x, class_map[(y, x)]) for (y, x) in annotation_points
    ]

    for patch in valid_patches:
        y_start, x_start, y_end, x_end = patch

        img = slide.read_region((x_start, y_start), 0, (patch_w, patch_h)).convert("RGB")
        img_path = os.path.join(images_dir, f"{image_id}_patch_{y_start}_{x_start}.jpg")
        img.save(img_path, "JPEG")

        rel_annots = get_relative_annotations_with_class(patch, annotations_with_class)
        txt_path = os.path.join(labels_dir, f"{image_id}_patch_{y_start}_{x_start}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for ann in rel_annots:
                f.write(f"{ann['class']} {ann['rel_x']} {ann['rel_y']}\n")

    slide.close()
    print(f"Extracted {len(valid_patches)} patches into {output_dir}")

    converter_para_yolo_paralelo(labels_dir, yolo_labels_dir, patch_w, patch_h)


if __name__ == "__main__":
    main()
