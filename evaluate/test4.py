import random

import numpy as np
from scipy import ndimage, misc
import matplotlib.pyplot as plt
from scipy.ndimage import generate_binary_structure

# fig = plt.figure()

# plt.gray()  # show the filtered result in grayscale
# ax1 = fig.add_subplot(121)  # left side
# ax2 = fig.add_subplot(122)  # right side
# ascent = misc.ascent()
# result = ndimage.maximum_filter(ascent, footprint=generate_binary_structure(2, 1)) * (ascent > 200)
# peaks_binary = (ndimage.maximum_filter(ascent, footprint=generate_binary_structure(2, 1)) == ascent) * (ascent > 200)
#
# print(peaks_binary)
# res = np.array(np.nonzero(peaks_binary)[::-1]).T
# print(res)
# print(len(res))
# ax1.imshow(ascent)
# ax2.imshow(peaks_binary)
# plt.show()

print(random.random())
