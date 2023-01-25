from time import time
from urllib.request import Request, urlopen
import asyncio
import os
import requests
import cv2, io, time
from PIL import Image
import torch
from tqdm import tqdm
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.transforms.functional import crop


def image_loader(image_path):
    img_cv = cv2.imread(image_path)
    img_pil = Image.open(image_path)

    binary_cv = cv2.imencode('.PNG', img_cv)[1].tobytes()

    return binary_cv


def main():
    images_path = "/home/user/Paint-by-Example/dataset/ohouse_images/raw"

    file_list = os.listdir(images_path)
    images_path = [os.path.join(images_path, file_name) for file_name in file_list]
    # images_path = images_path[]
    # file_list = file_list[]
    count_eliminated = 0
    for path, file_name in tqdm(zip(images_path, file_list), total=len(images_path)):
        # bboxes = []
        try:
            image = image_loader(path)
            result = get_object_detection_results(image)
            image = Image.open(io.BytesIO(image)).convert("RGB")
            cropped_images, bboxes, labels = get_image_bboxes(image, result)
        except:
            count_eliminated += 1
            continue
        if len(bboxes) > 0:
            with open("/home/user/Paint-by-Example/dataset/ohouse_images/annotation/" + file_name.rsplit(".", 1)[0] + '.txt', 'w',
                      encoding='utf-8') as f:
                for bbox, label in zip(bboxes, labels):
                    print(bbox[0], bbox[1], bbox[2], bbox[3], label, file=f)

    print(count_eliminated)

    # futures = [(get_object_detection_results(image_loader(path)), dfile_name) for path, file_name in
    #            zip(images_path, file_list)]
    #
    # for bboxes, file_name in futures:
    #     with open("/home/user/image_editing/dataset/lsun/annoations/" + file_name.split(".")[0] + '.txt', 'w', encoding='utf-8') as f:
    #         for bbox in bboxes:
    #             print(bbox[0], bbox[1], bbox[2], bbox[3], file=f)
    #
    # return futures


# async def main():
#     path = '/home/user/image_editing/dataset/lsun/raw/cbc27035dec1de99846267db82e3c90bfcf19841.jpg'
#     img_cv = cv2.imread(path)
#     img_pil = Image.open(path)
#
#     binary_cv = cv2.imencode('.PNG', img_cv)[1].tobytes()
#     futures = [asyncio.ensure_future(get_object_detection_results(binary_cv)) for i in range(10)]
#
#     result = await asyncio.gather(*futures)  # 결과를 한꺼번에 가져옴
#     print(result)


def get_object_detection_results(image_b):
    object_detection_api_url = "http://ac3bax2006.bdp.bdata.ai:11000"
    detection_results = requests.post(object_detection_api_url,
                                      files={"image": image_b}, params={"conf_th": 0.3}).json()["result"]

    return detection_results


class DetectionResult:
    def __init__(self, xmin, ymin, xmax, ymax, score=1.0, class_name=None):
        # Convert Tensor to python numbers.
        def maybe_itemize(val):
            if isinstance(val, torch.Tensor):
                val = val.item()
            return val

        xmin = maybe_itemize(xmin)
        ymin = maybe_itemize(ymin)
        xmax = maybe_itemize(xmax)
        ymax = maybe_itemize(ymax)

        # Sanity checks.
        ALMOST_ONE = 1.0001  # To handle float errors.
        if not all([0.0 <= xmin <= ALMOST_ONE, 0.0 <= ymin <= ALMOST_ONE,
                    0.0 <= xmax <= ALMOST_ONE, 0.0 <= ymax <= ALMOST_ONE]):
            raise ValueError("Bounding box must be given as relative coordinates (0.0 ~ 1.0).")

        if xmin > xmax or ymin > ymax:
            raise ValueError("Invalid coordinates.")

        self.xmin = max(0.0, xmin)
        self.ymin = max(0.0, ymin)
        self.xmax = min(1.0, xmax)
        self.ymax = min(1.0, ymax)
        self.score = score
        self.class_name = class_name

    @classmethod
    def from_abs_coords(cls, width, height, xmin, ymin, xmax, ymax, score=1.0, class_name=None):
        result = cls(
            xmin=xmin / width,
            ymin=ymin / height,
            xmax=xmax / width,
            ymax=ymax / height,
            score=score,
            class_name=class_name
        )

        return result

    # Return the box as a tuple of 4 numbers representing xmin, ymin, xmax, ymax.
    def coord(self, width=None, height=None, abs_coords=True):
        if abs_coords:
            assert width is not None and height is not None

            def to_abs(frac, cap):
                return min(round(frac * cap), cap - 1)

            return to_abs(self.xmin, width), to_abs(self.ymin, height), to_abs(self.xmax, width), to_abs(self.ymax,
                                                                                                         height)
        else:
            return self.xmin, self.ymin, self.xmax, self.ymax

    # Get a dictionary representation of this object.
    def to_dict(self):
        result = {
            "xmin": self.xmin,
            "ymin": self.ymin,
            "xmax": self.xmax,
            "ymax": self.ymax,
            "score": self.score,
            "class_name": self.class_name
        }

        return result


def get_image_bboxes(image, detection_results):
    detection_results = [
        DetectionResult(
            xmin=max(0, r["x"]),
            ymin=max(0, r["y"]),
            xmax=(max(0, r["x"]) + r["w"]),
            ymax=(max(0, r["y"]) + r["h"]),
            score=r["score"],
            class_name=r["class_name"],
        ) for r in detection_results
    ]
    img_int = torch.round(ToTensor()(image) * 255).to(torch.uint8)

    if img_int.shape[0] > 3:
        img_int = img_int[:3]
    img_width = img_int.shape[2]
    img_height = img_int.shape[1]

    boxes = [r.coord(img_width, img_height) for r in detection_results]
    labels = [r.class_name for r in detection_results]

    img_int = torch.round(ToTensor()(image) * 255).to(torch.uint8)
    crop_images = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        crop_images.append(crop(ToPILImage()(img_int), y_min, x_min, y_max - y_min, x_max - x_min))

    return crop_images, boxes, labels


if __name__ == "__main__":
    main()
