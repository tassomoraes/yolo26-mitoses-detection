import argparse

def get_patch_coordinates(image_dim, patch_dim):
    """
    Calculates the top-left and bottom-right coordinates of image patches.
    
    Args:
        image_dim (tuple): (image_height, image_width)
        patch_dim (tuple): (patch_height, patch_width)
        
    Returns:
        list of tuples: A list containing (y_start, x_start, y_end, x_end) for each patch.
    """
    img_h, img_w = image_dim
    patch_h, patch_w = patch_dim

    def get_1d_starts(length, p_length):
        if length <= p_length:
            return [0]
            
        starts = []
        current = 0
        
        while current + p_length <= length:
            starts.append(current)
            current += p_length
            
        if starts[-1] + p_length < length:
            starts.append(length - p_length)
            
        return starts

    y_starts = get_1d_starts(img_h, patch_h)
    x_starts = get_1d_starts(img_w, patch_w)

    coordinates = []
    for y in y_starts:
        for x in x_starts:
            y_end = y + patch_h
            x_end = x + patch_w
            coordinates.append((y, x, y_end, x_end))

    return coordinates

if __name__ == "__main__":
    # add_help=False previne que '-h' chame o menu de ajuda por padrão.
    parser = argparse.ArgumentParser(
        description="Calculate image patch coordinates.", 
        add_help=False
    )
    
    # Definindo os argumentos
    parser.add_argument('-w', '--width', type=int, required=True, help="Image width")
    parser.add_argument('-h', '--height', type=int, required=True, help="Image height")
    parser.add_argument('-pw', '--patch-width', type=int, required=True, help="Patch width")
    parser.add_argument('-ph', '--patch-height', type=int, required=True, help="Patch height")
    
    # Readicionando o menu de ajuda para o comando --help
    parser.add_argument('--help', action='help', help="Show this help message and exit")

    args = parser.parse_args()
    
    # Mapeando os argumentos para o formato esperado pela função (altura, largura)
    image_size = (args.height, args.width)
    patch_size = (args.patch_height, args.patch_width)
    
    coords = get_patch_coordinates(image_size, patch_size)
    
    # Exibindo os resultados
    print(f"Image Size: {image_size} | Patch Size: {patch_size}")
    print(f"Total patches: {len(coords)}")
    print("-" * 30)
    
    for i, box in enumerate(coords):
        print(f"Patch {i+1}: {box}")