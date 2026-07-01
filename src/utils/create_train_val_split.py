import argparse
import csv
import random
from pathlib import Path

def create_train_val_split(input_dir, output_csv, train_ratio=0.8, seed=42):
    """
    Reads .svs images from a folder, splits them into train and val sets,
    and saves the mapping to a CSV file.
    
    Args:
        input_dir (str): Path to the folder containing .svs images.
        output_csv (str): Path where the output CSV will be saved.
        train_ratio (float): Percentage of data to use for training (default 0.8 / 80%).
        seed (int): Random seed for reproducibility.
    """
    folder_path = Path(input_dir)
    
    # 1. Find all .svs files in the directory
    svs_files = list(folder_path.glob("*.svs"))
    
    if not svs_files:
        print(f"[-] No .svs files found in: {folder_path.resolve()}")
        return
        
    print(f"[+] Found {len(svs_files)} .svs files.")
    
    # Extract just the filenames (e.g., 'slide_01.svs')
    file_names = [f.name for f in svs_files]
    
    # 2. Shuffle the list to ensure a random split
    random.seed(seed)
    random.shuffle(file_names)
    
    # 3. Calculate the split index
    split_index = int(len(file_names) * train_ratio)
    
    train_files = file_names[:split_index]
    val_files = file_names[split_index:]
    
    # 4. Prepare the data for the CSV
    csv_data = []
    for name in train_files:
        csv_data.append([name, "train"])
        
    for name in val_files:
        csv_data.append([name, "val"])
        
    # 5. Write data to the CSV file
    try:
        with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # Write the header row
            writer.writerow(["filename", "split"])
            # Write the data rows
            writer.writerows(csv_data)
            
        print(f"[+] Split successfully applied:")
        print(f"    - Train set: {len(train_files)} images")
        print(f"    - Validation set: {len(val_files)} images")
        print(f"[+] CSV saved to: {Path(output_csv).resolve()}")
        
    except Exception as e:
        print(f"[-] Error writing to CSV file: {e}")

# --- Command Line Interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split .svs images into Train and Validation sets.")
    
    parser.add_argument('-i', '--input', type=str, required=True, 
                        help="Folder containing the .svs files")
    parser.add_argument('-o', '--output', type=str, required=True, 
                        help="Path and filename for the output CSV (e.g., dataset_split.csv)")
    parser.add_argument('-r', '--ratio', type=float, default=0.8, 
                        help="Ratio of images for the training set (default: 0.8 for 80%%)")
    parser.add_argument('-s', '--seed', type=int, default=42, 
                        help="Random seed for reproducible shuffles (default: 42)")

    args = parser.parse_args()
    
    create_train_val_split(
        input_dir=args.input, 
        output_csv=args.output, 
        train_ratio=args.ratio,
        seed=args.seed
    )