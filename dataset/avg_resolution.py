import PIL
from PIL import Image
import os
from tqdm.auto import tqdm

widths = []
heights = []

path = "/home/user/Paint-by-Example/dataset/ohouse_images/raw" # 265, 357
# path = "/home/user/image_editing/dataset/open-images/raw/train" # 787, 974

for img in tqdm(os.listdir(path)):
    img_path = os.path.join(path, img)  # Making image file path
    im = Image.open(img_path)
    widths.append(im.size[0])
    heights.append(im.size[1])

AVG_HEIGHT = round(sum(heights) / len(heights))
AVG_WIDTH = round(sum(widths) / len(widths))

print(AVG_HEIGHT, AVG_WIDTH)
