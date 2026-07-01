import concurrent.futures
from functools import partial
from coordinates import get_patch_coordinates

def _check_single_patch(patch, annotations):
    """
    Helper function to check a single patch. 
    This is separated so it can be easily distributed across CPU cores.
    """
    y_start, x_start, y_end, x_end = patch
    
    for py, px in annotations:
        if y_start <= py < y_end and x_start <= px < x_end:
            return patch  # Return the patch if it contains at least one point
            
    return None  # Return None if empty

def filter_patches_parallel(patches, annotations, max_workers=None):
    """
    Filters patch coordinates using multiprocessing, returning only the patches 
    that contain at least one annotation point.
    
    Args:
        patches (list of tuples): Format [(y_start, x_start, y_end, x_end), ...]
        annotations (list of tuples): Format [(y, x), ...] representing center points.
        max_workers (int, optional): Number of CPU cores to use. Defaults to all available.
        
    Returns:
        list of tuples: A filtered list of patches.
    """
    # functools.partial allows us to pass the static 'annotations' list to the helper function 
    # alongside the single 'patch' that the executor will map over.
    check_func = partial(_check_single_patch, annotations=annotations)
    
    valid_patches = []
    
    # ProcessPoolExecutor creates separate Python processes to bypass the GIL and utilize multiple CPU cores
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # executor.map distributes the patches across the available cores
        results = executor.map(check_func, patches)
        
    # Filter out the None results returned by empty patches
    valid_patches = [res for res in results if res is not None]
        
    return valid_patches

# --- Example Usage ---
if __name__ == "__main__":
    # 1. Define dimensions
    image_size = (1000, 1000)
    patch_size = (250, 256)
    
    # 2. Get patches by calling the imported function from coordinates.py
    all_patches = get_patch_coordinates(image_size, patch_size)
    
    # 3. Mock list of center annotations (y, x)
    mock_annotations = [
        (100, 100),   
        (300, 100),   
        (400, 150),
        (900, 800)    
    ]
    
    print(f"Total patches generated: {len(all_patches)}")
    print("Filtering in parallel...")
    
    # 4. Filter the patches using parallel processing
    filtered_patches = filter_patches_parallel(all_patches, mock_annotations)
    
    # Output the results
    print("-" * 30)
    print(f"Filtered patches count: {len(filtered_patches)}")
    for i, box in enumerate(filtered_patches):
        print(f"Kept Patch {i+1}: {box}")