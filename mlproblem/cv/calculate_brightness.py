def calculate_brightness(img: list[list[int]]) -> float:
    
    # Check if image is empty
    if not img or not img[0]:
        return -1
    # Check for inconsistent row lengths
    row_length = len(img[0])
    for row in img:
        if len(row) != row_length:
            return -1
    # Check for invalid pixel values
    for row in img:
        for pixel in row:
            if not (0<=pixel<=255):
                return -1
    # Calculate sum and total pixels
    total_pixels = row_length * len(img)
    total_brightness = sum(sum(row) for row in img)

    # Calculate average brightness
    avg_brightness = total_brightness / total_pixels
    return round(avg_brightness, 2)