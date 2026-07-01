# Augmentation Pipeline for YOLO Training
import os
import sys
import albumentations as A
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

# Fix np random seed to 42
np.random.seed(42)

# ─────────────────────────────────────────────
#  H&E color-space augmentation (Tellez et al.)
#  RGB → OD → H/E/R channels → perturb → RGB
# ─────────────────────────────────────────────

# Ruifrok & Johnston (2001) stain vectors for H&E
_HE_MATRIX = np.array([
    [0.65, 0.70, 0.29],   # Hematoxylin
    [0.07, 0.99, 0.11],   # Eosin
    [0.27, 0.57, 0.78],   # Residual
], dtype=np.float64)
 
_HE_INVERSE = np.linalg.inv(_HE_MATRIX)

# Fix np random seed to 42
np.random.seed(42)
 
 
def _he_color_augment(
    image: np.ndarray,
    alpha_range:      tuple = (0.95, 1.05),
    beta_range:       tuple = (-0.05, 0.05),
    brightness_limit: float = 0.20,
    contrast_limit:   float = 0.20,
    color_intensity_range: tuple = (0.0, 1.0),
) -> np.ndarray:
    """
    Full Tellez et al. 2018 stain augmentation pipeline (Figure 3).
    All perturbations are applied in the H&E decomposed space,
    BEFORE the channels are recombined back to RGB.
 
    Args:
        image                 : uint8 RGB input patch.
        alpha_range           : Multiplicative range per channel  [0.95, 1.05].
        beta_range            : Additive range per channel        [-0.05, 0.05].
        brightness_limit      : Max absolute brightness shift per channel.
        contrast_limit        : Max absolute contrast scale per channel.
        color_intensity_range : Range for color intensity blend factor λ.
                                λ=0 → full grayscale, λ=1 → full color.
    """
 
    # ── Step 1: RGB → optical density (OD) ──────────────────────────
    # Beer-Lambert law: OD = -log(I/I0), with I0=1 (normalised to [0,1])
    # Clip to avoid log(0); pixels at 0 are mapped to a large OD value.
    img = image.astype(np.float64) / 255.0
    img = np.clip(img, 1e-6, 1.0)
    od = -np.log(img)                           # shape (H, W, 3)
 
    # ── Step 2: OD → H / E / Residual channels (color deconvolution) ─
    # Each pixel's OD vector is projected onto the 3 stain basis vectors
    # using the pseudo-inverse of the Ruifrok stain matrix.
    h, w, _ = od.shape
    od_flat     = od.reshape(-1, 3)             # (N, 3)
    he_channels = od_flat @ _HE_INVERSE         # (N, 3): cols = H, E, R
 
    # ── Step 3: α · channel + β  (Tellez novel contribution) ────────
    # Each of the 3 channels is independently scaled and shifted with
    # random factors drawn from uniform distributions (paper ranges).
    alphas = np.random.uniform(alpha_range[0], alpha_range[1], size=3)
    betas  = np.random.uniform(beta_range[0],  beta_range[1],  size=3)
    he_channels = he_channels * alphas + betas  # broadcast over N pixels
 
    # ── Step 4: Brightness perturbation (per channel, in OD space) ──
    # A random additive shift is applied to each channel independently.
    # Working in OD space means this affects perceived stain darkness.
    brightness_shifts = np.random.uniform(
        -brightness_limit, brightness_limit, size=3
    )
    he_channels = he_channels + brightness_shifts
 
    # ── Step 5: Contrast perturbation (per channel, in OD space) ────
    # Scale each channel around its own mean, stretching or compressing
    # the dynamic range of that individual stain channel.
    contrast_scales = 1.0 + np.random.uniform(
        -contrast_limit, contrast_limit, size=3
    )
    channel_means   = he_channels.mean(axis=0)  # (3,) — one mean per channel
    he_channels     = (he_channels - channel_means) * contrast_scales + channel_means
 
    # ── Step 6: Color intensity blend (grayscale ↔ full-color) ──────
    # λ controls how much each channel keeps its individual identity:
    #   λ = 1 → original per-channel values (full color)
    #   λ = 0 → all channels set to their mean (grayscale in OD space)
    # This simulates images with reduced or exaggerated color contrast.
    lam = np.random.uniform(color_intensity_range[0], color_intensity_range[1])
    global_mean = he_channels.mean(axis=1, keepdims=True)  # (N, 1) per pixel
    he_channels = lam * he_channels + (1.0 - lam) * global_mean
 
    # Clip to non-negative OD values (negative OD has no physical meaning)
    he_channels = np.clip(he_channels, 0, None)
 
    # ── Step 7: Recombine channels → OD → RGB ───────────────────────
    # Project the modified H/E/R channels back to OD using the stain matrix,
    # then invert OD to recover the RGB intensities.
    od_reconstructed = he_channels @ _HE_MATRIX     # (N, 3) OD
    od_reconstructed = np.clip(od_reconstructed, 0, None)
 
    rgb = np.exp(-od_reconstructed)                  # invert Beer-Lambert
    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb.reshape(h, w, 3) * 255).astype(np.uint8)
 
 
class HEColorAugment(A.ImageOnlyTransform):
    """
    Albumentations wrapper for the full Tellez et al. 2018 stain pipeline.
    Brightness, contrast and color intensity are applied in H&E decomposed
    space — before channel recombination — as described in Figure 3.
    """
 
    def __init__(self,
                 alpha_range:           tuple = (0.95, 1.05),
                 beta_range:            tuple = (-0.05, 0.05),
                 brightness_limit:      float = 0.20,
                 contrast_limit:        float = 0.20,
                 color_intensity_range: tuple = (0.5, 1.0),
                 p: float = 1.0):
        super().__init__(p=p)
        self.alpha_range           = alpha_range
        self.beta_range            = beta_range
        self.brightness_limit      = brightness_limit
        self.contrast_limit        = contrast_limit
        self.color_intensity_range = color_intensity_range
 
    def apply(self, img, **params):
        return _he_color_augment(
            img,
            alpha_range=self.alpha_range,
            beta_range=self.beta_range,
            brightness_limit=self.brightness_limit,
            contrast_limit=self.contrast_limit,
            color_intensity_range=self.color_intensity_range,
        )
 
    def get_transform_init_args_names(self):
        return (
            "alpha_range",
            "beta_range",
            "brightness_limit",
            "contrast_limit",
            "color_intensity_range",
        )
 

# Define the augmentation pipeline
def get_augmentation_pipeline(
        R_p: float = 0.8, 
        S_p: float = 0.8, 
        E_p: float = 0.8, 
        B_p: float = 0.3,
        G_p: float = 0.3, 
        HE_p: float = 0.5
    ) -> A.Compose:
    
    return A.Compose([
        A.Rotate(
            limit=(-180, 180), 
            p=R_p,
            border_mode=0,         # cv2.BORDER_CONSTANT
            fill=(255, 255, 255)  # fundo branco para a imagem
        ),
        
        A.RandomScale(scale_limit=(-0.2, 0.2), p=S_p),
        
        A.ElasticTransform(alpha=1.0, sigma=50, p=E_p),
        
        A.Blur(blur_limit=(3, 5), p=B_p),
        
        A.GaussNoise(var_limit=(0.01, 0.05), p=G_p),

    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.4,
    ), seed=42)

def apply_augmentation(image, bboxes, class_labels):
    augmentation_pipeline = get_augmentation_pipeline()
    augmented = augmentation_pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
    return augmented['image'], augmented['bboxes'], augmented['class_labels']

def process_image_augmentations(image_data):
    """
    Process a single image and generate all its augmentations.
    This function is designed to be run in parallel.
    
    Args:
        image_data: dict with image_file, image_path, bboxes_path, output_images, output_labels
    
    Returns:
        tuple: (success, message)
    """
    try:
        image_file = image_data['image_file']
        image_path = image_data['image_path']
        bboxes_path = image_data['bboxes_path']
        output_images = image_data['output_images']
        output_labels = image_data['output_labels']
        
        image_name = os.path.splitext(image_file)[0]
        
        # Skip if corresponding label file doesn't exist
        if not os.path.exists(bboxes_path):
            return False, f"Warning: Label file not found for {image_file}, skipping..."
        
        # Load image and bounding boxes
        image = cv2.imread(image_path)
        if image is None:
            return False, f"Warning: Could not load image {image_file}, skipping..."
        
        bboxes = []
        class_labels = []
        with open(bboxes_path, 'r') as f:
            for line in f:
                values = line.strip().split()
                if not values:
                    continue
                class_labels.append(int(values[0]))       
                bboxes.append([
                    float(values[1]),  # x_center
                    float(values[2]),  # y_center
                    float(values[3]),  # width
                    float(values[4]),  # height
                ])
        
        # Apply augmentation 2 times
        for n in range(2):
            augmented_image, augmented_bboxes, augmented_class_labels = apply_augmentation(image, bboxes, class_labels)
            
            # Save image and labels in YOLO format to disk
            output_image_path = os.path.join(output_images, f"{image_name}_aug_{n}.jpg")
            cv2.imwrite(output_image_path, augmented_image)
            
            output_label_path = os.path.join(output_labels, f"{image_name}_aug_{n}.txt")
            with open(output_label_path, 'w') as f:
                for bbox, label in zip(augmented_bboxes, augmented_class_labels):
                    f.write(f"{int(label)} {' '.join(map(str, bbox))}\n")
        
        return True, f"Processed {image_file}"
    
    except Exception as e:
        return False, f"Error processing {image_file}: {str(e)}"

def main():
    # Get images folder path from command line argument
    if len(sys.argv) < 2:
        print("Usage: python augmentation_pipeline.py <images_folder> [num_workers]")
        print("Example: python augmentation_pipeline.py data/MIDOGpp/images/train 8")
        sys.exit(1)
    
    images_folder = sys.argv[1]
    
    # Get number of workers (default to CPU count)
    num_workers = int(sys.argv[2]) if len(sys.argv) > 2 else os.cpu_count() or 4
    
    # Validate that the folder exists
    if not os.path.isdir(images_folder):
        print(f"Error: Folder '{images_folder}' does not exist.")
        sys.exit(1)
    
    # Determine paths based on the images folder
    # Assume structure: data/<dataset>/images/<split>/ and data/<dataset>/labels/<split>/
    label_path = images_folder.replace("images", "labels")
    if not os.path.isdir(label_path):
        print(f"Error: Corresponding labels folder '{label_path}' does not exist.")
        sys.exit(1)
    
    images_path = Path(images_folder)
    labels_folder = Path(label_path)
    output_base = "data/MIDOGpp/augmented_data"
    output_images = os.path.join(output_base, "images")
    output_labels = os.path.join(output_base, "labels")
    
    # Create output directories if they don't exist
    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_labels, exist_ok=True)
    
    # Process all image files in the folder
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in os.listdir(images_folder) 
                   if os.path.isfile(os.path.join(images_folder, f)) 
                   and os.path.splitext(f)[1].lower() in image_extensions]
    
    if not image_files:
        print(f"No images found in {images_folder}")
        sys.exit(0)
    
    print(f"Found {len(image_files)} images in {images_folder}")
    print(f"Using {num_workers} workers for parallel processing\n")
    
    # Prepare data for parallel processing
    tasks = []
    for image_file in image_files:
        image_name = os.path.splitext(image_file)[0]
        image_path = os.path.join(images_folder, image_file)
        labels_file = f"{image_name}.txt"
        bboxes_path = os.path.join(labels_folder, labels_file)
        
        tasks.append({
            'image_file': image_file,
            'image_path': image_path,
            'bboxes_path': bboxes_path,
            'output_images': output_images,
            'output_labels': output_labels,
        })
    
    # Process images in parallel
    completed = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(process_image_augmentations, task): task for task in tasks}
        
        # Process completed tasks as they finish
        for future in as_completed(future_to_task):
            completed += 1
            success, message = future.result()
            
            if success:
                print(f"[{completed}/{len(image_files)}] {message}")
            else:
                failed += 1
                print(f"[{completed}/{len(image_files)}] {message}")
    
    print(f"\n✓ Processing complete!")
    print(f"  - Successfully processed: {completed - failed}")
    print(f"  - Failed/Skipped: {failed}")

if __name__ == "__main__":
    main()