import numpy as np


# Пример расчета сверточной сети

convolution_with_padding = np.array([
                            [0, 0, 0],
                            [0, 5, 2],
                            [0, 3, 1]
                            ])

convolution_with_padding2 = np.array([
                            [0, 0, 0],
                            [5, 2, 3],
                            [3, 1, 1]
                            ])

convolution_with_padding3 = np.array([
                            [0, 0, 0],
                            [2, 3, 1],
                            [1, 1, 0]
                            ])

convolution_kernel = np.array([
                            [1, 0, -1],
                            [0, 1, 0],
                            [-1, 0, 1]
                            ])

convolution_with_padding_ex = np.array([
                                    [4, 2, -1],
                                    [-6, 0, 5],
                                    [3, 2, 2]
                                    ])

convolution_kernel2 = np.array([
                            [0, 1, 2],
                            [1, -1, 0],
                            [1, 0, -2]
                            ])

multiply1 = (convolution_with_padding * convolution_kernel).sum()
multiply2 = (convolution_with_padding2 * convolution_kernel).sum()
multiply3 = (convolution_with_padding3 * convolution_kernel).sum()
multiply4 = (convolution_with_padding_ex * convolution_kernel2).sum()

print(multiply1)
print(multiply2)
print(multiply3)
