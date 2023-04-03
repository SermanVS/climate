import math
def conv2d_size_out(size, kernel_size, stride, padding=0, dilation=1):
    return math.floor((size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

def convtranspose2d_size_out(size, kernel_size, stride, padding=0, dilation=1, output_padding=0):
    return (size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1

def maxpool2d_size_out(size, kernel_size=2, stride=2):
    return math.floor((size - (kernel_size - 1) - 1) / stride + 1)