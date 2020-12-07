# Kernels
# 1.Kernels can be an arbitrary size of M * N pixels
# M and N are odd integers
# to ensure valid (x, y)-coordinate at the center of the kernel
# 2.slides from left-to-right and top-to-bottom


# Convolution
# 1.Select an (x, y)-coordinate from the original image.
# 2.Place the center of the kernel at this (x, y) coordinate.
# 3.Multiply each kernel value by the corresponding input image pixel value
#   and then take the sum of all multiplication operations.
# 4.Use the same (x, y)-coordinate from Step 1
#   store the kernel output in the same (x, y)-location as the output image.
