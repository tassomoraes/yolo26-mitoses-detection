import json

def generate_relative_annotations_json(patches, annotations):
    """
    Transforms global annotation coordinates to relative coordinates for each patch 
    and outputs the result as a JSON string.
    
    Args:
        patches (list of tuples): Format [(y_start, x_start, y_end, x_end), ...]
        annotations (list of tuples): Format [(y, x), ...] (global coordinates)
        
    Returns:
        str: A JSON formatted string containing the patches and their relative annotations.
    """
    registry = []
    
    for patch in patches:
        y_start, x_start, y_end, x_end = patch
        relative_annots = []
        
        # Find annotations inside this patch and convert to relative coordinates
        for py, px in annotations:
            if y_start <= py < y_end and x_start <= px < x_end:
                # Calculate relative position
                rel_y = py - y_start
                rel_x = px - x_start
                
                relative_annots.append({
                    "y": rel_y, 
                    "x": rel_x
                })
        
        # Only add to the registry if the patch actually contains annotations
        if relative_annots:
            patch_data = {
                "patch_bbox": {
                    "y_start": y_start, 
                    "x_start": x_start, 
                    "y_end": y_end, 
                    "x_end": x_end
                },
                "annotations": relative_annots
            }
            registry.append(patch_data)
            
    # Convert the Python dictionary to a formatted JSON string
    return json.dumps(registry, indent=4)


def get_relative_annotations_with_class(patch, annotations_with_class):
    """
    Returns annotations with coordinates relative to the patch and class label.

    Args:
        patch (tuple): (y_start, x_start, y_end, x_end)
        annotations_with_class (list): [(y, x, class_label), ...]
            where class_label is "mitose" or "nao_mitose"

    Returns:
        list of dict: [{"class": label, "rel_x": int, "rel_y": int}, ...]
            only for annotations contained in the patch
    """
    y_start, x_start, y_end, x_end = patch
    result = []
    for py, px, label in annotations_with_class:
        if y_start <= py < y_end and x_start <= px < x_end:
            result.append({
                "class": label,
                "rel_x": px - x_start,
                "rel_y": py - y_start
            })
    return result


# --- Example Usage ---
if __name__ == "__main__":
    # 1. Mock list of patches (e.g., from the filter function)
    mock_patches = [
        (0, 0, 250, 256),     # Patch 1: Top-Left
        (250, 0, 500, 256),   # Patch 2: Bottom-Left
    ]
    
    # 2. Mock list of global center annotations (y, x)
    mock_annotations = [
        (100, 100),  # Inside Patch 1
        (300, 100),  # Inside Patch 2
        (400, 150)   # Inside Patch 2
    ]
    
    # 3. Generate JSON
    json_output = generate_relative_annotations_json(mock_patches, mock_annotations)
    
    # Output the results
    print("Generated JSON Registry:")
    print(json_output)
    
    # Optional: If you want to save it to a file
    # with open("annotations_registry.json", "w") as f:
    #     f.write(json_output)