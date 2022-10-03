import math
def conv2d_size_out(size, kernel_size=5, stride=2):
    """
    common use case:
    cur_layer_img_w = conv2d_size_out(cur_layer_img_w, kernel_size, stride)
    cur_layer_img_h = conv2d_size_out(cur_layer_img_h, kernel_size, stride)
    to understand the shape for dense layer's input
    """
    return (size - (kernel_size - 1) - 1) // stride  + 1

def maxpool2d_size_out(size, kernel_size=2, stride=2):
    return math.floor((size - (kernel_size - 1) - 1) / stride + 1)