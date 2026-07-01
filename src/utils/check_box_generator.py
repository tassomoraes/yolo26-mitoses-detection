import os
import tifffile

def load_tiff_image(image_path):
    # load tiff image using tifffile
    return tifffile.imread(image_path)


def visualize_checkbox_on_tiff():
    for image_path in os.listdir("data/MIDOGpp/images/train"):
        label_path = image_path.replace("images", "labels").replace(".jpg", ".txt")
        
        # load tiff image and visualize checkbox
        image = load_tiff_image(f"data/MIDOGpp/images/train/{image_path}")

        # aply visualization of checkbox on the image using the label file


def main():
    for image_path in os.listdir("data/MIDOGpp/images/train"):
        label_path = image_path.replace("images", "labels").replace(".jpg", ".txt")
        visualize_tile(
            f"data/MIDOGpp/images/train/{image_path}",
            f"data/MIDOGpp/labels/train/{label_path}",
            f"checkbox_imgs/{image_path.replace('.jpg', '.png')}",
        )

if __name__ == "__main__":
    main()