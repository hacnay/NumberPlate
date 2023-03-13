import numpy as np
from skimage.transform import resize
from skimage import measure
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import DetectPlate

# The invert was done so as to convert the black pixel to white pixel and vice versa
license_plate = np.invert(DetectPlate.plate_like_objects[0])

labelled_plate = measure.label(license_plate)

fig, ax1 = plt.subplots(1)
ax1.imshow(license_plate, cmap="gray")
# the next two lines is based on the assumptions that the width of
# a license plate should be between 5% and 15% of the license plate,
# and height should be between 35% and 60%
# this will eliminate some
min_height, max_height = 0.35 * license_plate.shape[0], 0.6 * license_plate.shape[0]
min_width, max_width = 0.05 * license_plate.shape[1], 0.15 * license_plate.shape[1]

characters = []
column_list = []
for regions in measure.regionprops(labelled_plate):
    y0, x0, y1, x1 = regions.bbox
    region_height, region_width = y1 - y0, x1 - x0

    if min_height < region_height < max_height and min_width < region_width < max_width:
        roi = license_plate[y0:y1, x0:x1]

        # draw a red bordered rectangle over the character.
        rect_border = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="red", linewidth=2, fill=False)
        ax1.add_patch(rect_border)

        # resize the characters to 20X20 and then append each character into the characters list
        resized_char = resize(roi, (20, 20))
        characters.append(resized_char)

        # this is just to keep track of the arrangement of the characters
        column_list.append(x0)

plt.show()
