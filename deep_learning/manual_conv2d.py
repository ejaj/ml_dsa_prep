import numpy as np

def conv2d_manual(image, kernel, stride=1, padding=0):
    """
    Manually performs 2D convolution on a grayscale image.
    
    Args:
    - image (2D numpy array): input image (H x W)
    - kernel (2D numpy array): convolution kernel (kH x kW)
    - stride (int): stride of the convolution
    - padding (int): zero-padding around the image
    
    Returns:
    - output (2D numpy array): convolved feature map
    """
    H, W = image.shape
    kH, kW = kernel.shape
    padded_image = np.pad(image, pad_width=padding, mode='constant', constant_values=0)
    # Output size
    out_height = ((H + 2 * padding - kH) // stride) + 1
    out_width = ((W + 2 * padding - kW) // stride) + 1
    output = np.zeros((out_height, out_width))

    # Flip the kernel (for true convolution)
    kernel_flipped = np.flipud(np.fliplr(kernel))
    
    # Perform convolution using nested loops
    for y in range(0, out_height):
        for x in range(0, out_width):
            y_start = y * stride
            y_end = y_start + kH
            x_start = x * stride
            x_end = x_start + kW

            region = padded_image[y_start:y_end, x_start:x_end]
            output[y, x] = np.sum(region * kernel_flipped)
    return output
sample_image = np.array([
    [1, 2, 3, 0, 1],
    [0, 1, 2, 3, 1],
    [3, 1, 0, 1, 2],
    [2, 0, 1, 3, 1],
    [1, 2, 3, 1, 0]
])

sample_kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

conv_output = conv2d_manual(sample_image, sample_kernel, stride=1, padding=1)
conv_output