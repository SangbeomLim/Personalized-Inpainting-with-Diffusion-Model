import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageDraw
from transformers import CLIPProcessor
import random
import torchvision
import albumentations as A
import cv2
import copy
import bezier
import torch
import torchvision.transforms as T

def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                            (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                            (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)

imagenet_templates_smallest = [
    'a photo of a {}',
]

class LsunPretrainImageTextInversion(Dataset):
    def __init__(self, state, arbitrary_mask_percent=0, dataset_ratio=(0.9, 0.05, 0.05), class_label_exist=False, **args
                 ):
        self.state = state
        self.args = args
        self.arbitrary_mask_percent = arbitrary_mask_percent
        self.kernel = np.ones((1, 1), np.uint8)
        self.random_trans = A.Compose([
            A.Resize(height=224, width=224),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20),
            A.Blur(p=0.3),
            A.ElasticTransform(p=0.3)
        ])
        self.dataset_ratio = dataset_ratio
        self.class_label_exist = class_label_exist

        class_dict = [
            'null',
            'home-appliance',
            'decoration',
            'cabinet_shelf',
            'door',
            'table',
            'lamp',
            'plant',
            'chair_stool',
            'curtain',
            'sofa',
            'bed'
        ]
        self.class_dict = {v: k for k, v in enumerate(class_dict)}

        bad_list = [
            '1af17f3d912e9aac.txt',
            '1d5ef05c8da80e31.txt',
            '3095084b358d3f2d.txt',
            '3ad7415a11ac1f5e.txt',
            '42a30d8f8fba8b40.txt',
            '1366cde3b480a15c.txt',
            '03a53ed6ab408b9f.txt'
        ]
        self.bbox_path_list = []
        if state == "train":
            # dir_name_list=['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f']
            # for dir_name in dir_name_list:
            #     bbox_dir=os.path.join(args['dataset_dir'],'bbox','train_'+dir_name)
            bbox_dir = os.path.join(args['dataset_dir'], 'annotations')
            per_dir_file_list = os.listdir(bbox_dir)
            per_dir_file_list=per_dir_file_list[:int(len(per_dir_file_list)*self.dataset_ratio[0])] # 90%
            for file_name in per_dir_file_list:
                if file_name not in bad_list:
                    self.bbox_path_list.append(os.path.join(bbox_dir, file_name))
        elif state == "validation":
            bbox_dir = os.path.join(args['dataset_dir'], 'annotations')
            per_dir_file_list = os.listdir(bbox_dir)
            start_index = int(len(per_dir_file_list) * self.dataset_ratio[0])
            end_index = int(len(per_dir_file_list) * self.dataset_ratio[1]) + start_index
            per_dir_file_list = per_dir_file_list[start_index:end_index] # 5%
            for file_name in per_dir_file_list:
                if file_name not in bad_list:
                    self.bbox_path_list.append(os.path.join(bbox_dir, file_name))
        else:
            bbox_dir = os.path.join(args['dataset_dir'], 'annotations')
            per_dir_file_list = os.listdir(bbox_dir)
            start_index = int(len(per_dir_file_list) * self.dataset_ratio[0]) + int(
                len(per_dir_file_list) * self.dataset_ratio[1])
            end_index = int(len(per_dir_file_list) * self.dataset_ratio[1]) + start_index
            per_dir_file_list = per_dir_file_list[start_index:end_index] # 5%
            for file_name in per_dir_file_list:
                if file_name not in bad_list:
                    self.bbox_path_list.append(os.path.join(bbox_dir, file_name))
        self.bbox_path_list.sort()
        self.length = len(self.bbox_path_list)

    def __getitem__(self, index):
        bbox_path = self.bbox_path_list[index]
        file_name = os.path.splitext(os.path.basename(bbox_path))[:-1] + '.jpg'
        # dir_name = bbox_path.split('/')[-2]
        img_path = os.path.join('/home/user/Paint-by-Example/dataset/ohouse_images/raw', file_name)  # Path setting

        bbox_list = []
        label_list = []
        with open(bbox_path) as f:
            line = f.readline()
            while line:
                line_split = line.strip('\n').split(" ")
                bbox_temp = []
                for i in range(4):  # BBox 만 가져감, [4] 는 Label
                    bbox_temp.append(int(float(line_split[i])))
                bbox_list.append(bbox_temp)
                if self.class_label_exist:
                    label_list.append(line_split[4])  # Label 값 가져오기
                else:
                    label_list.append('null')
                line = f.readline()  # 다음 행
        bbox_index = random.randint(0, len(bbox_list) - 1)
        bbox = bbox_list[bbox_index]
        class_label = self.class_dict[label_list[bbox_index]]
        text = random.choice(imagenet_templates_smallest).format(label_list[bbox_index])
        # class_label = self.class_dict[label_list[bbox_index]]
        # bbox = random.choice(bbox_list)
        img_p = Image.open(img_path).convert("RGB")

        ### Get reference image
        bbox_pad = copy.copy(bbox)
        bbox_pad[0] = bbox[0] - min(10, bbox[0] - 0)
        bbox_pad[1] = bbox[1] - min(10, bbox[1] - 0)
        bbox_pad[2] = bbox[2] + min(10, img_p.size[0] - bbox[2])
        bbox_pad[3] = bbox[3] + min(10, img_p.size[1] - bbox[3])
        img_p_np = cv2.imread(img_path)
        img_p_np = cv2.cvtColor(img_p_np, cv2.COLOR_BGR2RGB)
        ref_image_tensor = img_p_np[bbox_pad[1]:bbox_pad[3], bbox_pad[0]:bbox_pad[2], :]
        ref_image_tensor = self.random_trans(image=ref_image_tensor)
        # try:
        #     ref_image_tensor = self.random_trans(image=ref_image_tensor)
        # except:
        #     with open("/home/user/Paint-by-Example/dataset/lsun/error_file.txt","a",encoding="utf-8") as f:
        #         f.write(file_name+'\n')
        ref_image_tensor = Image.fromarray(ref_image_tensor["image"])
        ref_image_tensor = get_tensor_clip()(ref_image_tensor)

        ### Generate mask
        image_tensor = get_tensor()(img_p)
        W, H = img_p.size

        extended_bbox = copy.copy(bbox)
        left_freespace = bbox[0] - 0
        right_freespace = W - bbox[2]
        up_freespace = bbox[1] - 0
        down_freespace = H - bbox[3]
        try:
            extended_bbox[0] = bbox[0] - random.randint(0, int(0.4 * left_freespace))
        except:
            extended_bbox[0] = bbox[0]
        try:  # Exception Added
            extended_bbox[1] = bbox[1] - random.randint(0, int(0.4 * up_freespace))
        except:
            extended_bbox[1] = bbox[1]
        try:
            extended_bbox[2] = bbox[2] - random.randint(0, int(0.4 * right_freespace))
        except:
            extended_bbox[2] = bbox[2]
        try:
            extended_bbox[3] = bbox[3] + random.randint(0, int(0.4 * down_freespace))  # Error Occured
        except:
            extended_bbox[3] = bbox[3]  # Error Occured

        prob = random.uniform(0, 1)
        if prob < self.arbitrary_mask_percent:
            mask_img = Image.new('RGB', (W, H), (255, 255, 255))
            bbox_mask = copy.copy(bbox)
            extended_bbox_mask = copy.copy(extended_bbox)
            top_nodes = np.asfortranarray([
                [bbox_mask[0], (bbox_mask[0] + bbox_mask[2]) / 2, bbox_mask[2]],
                [bbox_mask[1], extended_bbox_mask[1], bbox_mask[1]],
            ])
            down_nodes = np.asfortranarray([
                [bbox_mask[2], (bbox_mask[0] + bbox_mask[2]) / 2, bbox_mask[0]],
                [bbox_mask[3], extended_bbox_mask[3], bbox_mask[3]],
            ])
            left_nodes = np.asfortranarray([
                [bbox_mask[0], extended_bbox_mask[0], bbox_mask[0]],
                [bbox_mask[3], (bbox_mask[1] + bbox_mask[3]) / 2, bbox_mask[1]],
            ])
            right_nodes = np.asfortranarray([
                [bbox_mask[2], extended_bbox_mask[2], bbox_mask[2]],
                [bbox_mask[1], (bbox_mask[1] + bbox_mask[3]) / 2, bbox_mask[3]],
            ])
            top_curve = bezier.Curve(top_nodes, degree=2)
            right_curve = bezier.Curve(right_nodes, degree=2)
            down_curve = bezier.Curve(down_nodes, degree=2)
            left_curve = bezier.Curve(left_nodes, degree=2)
            curve_list = [top_curve, right_curve, down_curve, left_curve]
            pt_list = []
            random_width = 5
            for curve in curve_list:
                x_list = []
                y_list = []
                for i in range(1, 19):
                    if (curve.evaluate(i * 0.05)[0][0]) not in x_list and (
                            curve.evaluate(i * 0.05)[1][0] not in y_list):
                        pt_list.append((curve.evaluate(i * 0.05)[0][0] + random.randint(-random_width, random_width),
                                        curve.evaluate(i * 0.05)[1][0] + random.randint(-random_width, random_width)))
                        x_list.append(curve.evaluate(i * 0.05)[0][0])
                        y_list.append(curve.evaluate(i * 0.05)[1][0])
            mask_img_draw = ImageDraw.Draw(mask_img)
            mask_img_draw.polygon(pt_list, fill=(0, 0, 0))
            mask_tensor = get_tensor(normalize=False, toTensor=True)(mask_img)[0].unsqueeze(0)
        else:
            mask_img = np.zeros((H, W))
            mask_img[extended_bbox[1]:extended_bbox[3], extended_bbox[0]:extended_bbox[2]] = 1
            mask_img = Image.fromarray(mask_img)
            mask_tensor = 1 - get_tensor(normalize=False, toTensor=True)(mask_img)

        ### Crop square image
        if W > H:
            left_most = extended_bbox[2] - H
            if left_most < 0:
                left_most = 0
            right_most = extended_bbox[0] + H
            if right_most > W:
                right_most = W
            right_most = right_most - H
            if right_most <= left_most:
                image_tensor_cropped = image_tensor
                mask_tensor_cropped = mask_tensor
            else:
                left_pos = random.randint(left_most, right_most)
                free_space = min(extended_bbox[1] - 0, extended_bbox[0] - left_pos, left_pos + H - extended_bbox[2],
                                 H - extended_bbox[3])
                try:
                    random_free_space = random.randint(0, int(0.6 * free_space))
                except:
                    random_free_space = 0
                image_tensor_cropped = image_tensor[:, 0 + random_free_space:H - random_free_space,
                                       left_pos + random_free_space:left_pos + H - random_free_space]
                mask_tensor_cropped = mask_tensor[:, 0 + random_free_space:H - random_free_space,
                                      left_pos + random_free_space:left_pos + H - random_free_space]

        elif W < H:
            upper_most = extended_bbox[3] - W
            if upper_most < 0:
                upper_most = 0
            lower_most = extended_bbox[1] + W
            if lower_most > H:
                lower_most = H
            lower_most = lower_most - W
            if lower_most <= upper_most:
                image_tensor_cropped = image_tensor
                mask_tensor_cropped = mask_tensor
            else:
                upper_pos = random.randint(upper_most, lower_most)
                free_space = min(extended_bbox[1] - upper_pos, extended_bbox[0] - 0, W - extended_bbox[2],
                                 upper_pos + W - extended_bbox[3])
                try:
                    random_free_space = random.randint(0, int(0.6 * free_space))
                except:
                    random_free_space = 0
                image_tensor_cropped = image_tensor[:, upper_pos + random_free_space:upper_pos + W - random_free_space,
                                       random_free_space:W - random_free_space]
                mask_tensor_cropped = mask_tensor[:, upper_pos + random_free_space:upper_pos + W - random_free_space,
                                      random_free_space:W - random_free_space]
        else:
            image_tensor_cropped = image_tensor
            mask_tensor_cropped = mask_tensor

        image_tensor_resize = T.Resize([self.args['image_size'], self.args['image_size']])(image_tensor_cropped)
        mask_tensor_resize = T.Resize([self.args['image_size'], self.args['image_size']])(mask_tensor_cropped)
        inpaint_tensor_resize = image_tensor_resize * mask_tensor_resize

        return {"GT": image_tensor_resize, "inpaint_image": inpaint_tensor_resize, "inpaint_mask": mask_tensor_resize,
                "ref_imgs": ref_image_tensor, "caption": text, "class_label": torch.tensor(class_label)}

    def __len__(self):
        return self.length


    def test_get_item(self, index):
        bbox_path = self.bbox_path_list[index]
        file_name = os.path.splitext(os.path.basename(bbox_path))[0] + '.jpg'
        # dir_name = bbox_path.split('/')[-2]
        img_path = os.path.join('/home/user/Paint-by-Example/dataset/lsun/raw', file_name)  # Path setting

        bbox_list = []
        label_list = []
        with open(bbox_path) as f:
            line = f.readline()
            while line:
                line_split = line.strip('\n').split(" ")
                bbox_temp = []
                for i in range(4):  # BBox 만 가져감, [4] 는 Label
                    bbox_temp.append(int(float(line_split[i])))
                bbox_list.append(bbox_temp)
                if self.class_label_exist:
                    label_list.append(line_split[4])  # Label 값 가져오기
                else:
                    label_list.append('null')
                line = f.readline()  # 다음 행
        bbox_index = random.randint(0, len(bbox_list) - 1)
        bbox = bbox_list[bbox_index]
        class_label = self.class_dict[label_list[bbox_index]]
        text = random.choice(imagenet_templates_smallest).format(label_list[bbox_index])

        img_p = Image.open(img_path).convert("RGB")

        ### Get reference image
        bbox_pad = copy.copy(bbox)
        bbox_pad[0] = bbox[0] - min(10, bbox[0] - 0)
        bbox_pad[1] = bbox[1] - min(10, bbox[1] - 0)
        bbox_pad[2] = bbox[2] + min(10, img_p.size[0] - bbox[2])
        bbox_pad[3] = bbox[3] + min(10, img_p.size[1] - bbox[3])
        img_p_np = cv2.imread(img_path)
        img_p_np = cv2.cvtColor(img_p_np, cv2.COLOR_BGR2RGB)
        ref_image = img_p_np[bbox_pad[1]:bbox_pad[3], bbox_pad[0]:bbox_pad[2], :]

        ### Generate mask
        W, H = img_p.size

        extended_bbox = copy.copy(bbox)
        left_freespace = bbox[0] - 0
        right_freespace = W - bbox[2]
        up_freespace = bbox[1] - 0
        down_freespace = H - bbox[3]

        try:
            extended_bbox[0] = bbox[0] - random.randint(0, int(0.4 * left_freespace))
        except:
            extended_bbox[0] = bbox[0]
        try:  # Exception Added
            extended_bbox[1] = bbox[1] - random.randint(0, int(0.4 * up_freespace))
        except:
            extended_bbox[1] = bbox[1]
        try:
            extended_bbox[2] = bbox[2] - random.randint(0, int(0.4 * right_freespace))
        except:
            extended_bbox[2] = bbox[2]
        try:
            extended_bbox[3] = bbox[3] + random.randint(0, int(0.4 * down_freespace))  # Error Occured
        except:
            extended_bbox[3] = bbox[3]  # Error Occured

        mask_img = Image.new('RGB', (W, H), (255, 255, 255))
        bbox_mask = copy.copy(bbox)
        extended_bbox_mask = copy.copy(extended_bbox)
        top_nodes = np.asfortranarray([
            [bbox_mask[0], (bbox_mask[0] + bbox_mask[2]) / 2, bbox_mask[2]],
            [bbox_mask[1], extended_bbox_mask[1], bbox_mask[1]],
        ])
        down_nodes = np.asfortranarray([
            [bbox_mask[2], (bbox_mask[0] + bbox_mask[2]) / 2, bbox_mask[0]],
            [bbox_mask[3], extended_bbox_mask[3], bbox_mask[3]],
        ])
        left_nodes = np.asfortranarray([
            [bbox_mask[0], extended_bbox_mask[0], bbox_mask[0]],
            [bbox_mask[3], (bbox_mask[1] + bbox_mask[3]) / 2, bbox_mask[1]],
        ])
        right_nodes = np.asfortranarray([
            [bbox_mask[2], extended_bbox_mask[2], bbox_mask[2]],
            [bbox_mask[1], (bbox_mask[1] + bbox_mask[3]) / 2, bbox_mask[3]],
        ])
        top_curve = bezier.Curve(top_nodes, degree=2)
        right_curve = bezier.Curve(right_nodes, degree=2)
        down_curve = bezier.Curve(down_nodes, degree=2)
        left_curve = bezier.Curve(left_nodes, degree=2)
        curve_list = [top_curve, right_curve, down_curve, left_curve]
        pt_list = []
        random_width = 5
        for curve in curve_list:
            x_list = []
            y_list = []
            for i in range(1, 19):
                if (curve.evaluate(i * 0.05)[0][0]) not in x_list and (
                        curve.evaluate(i * 0.05)[1][0] not in y_list):
                    pt_list.append((curve.evaluate(i * 0.05)[0][0] + random.randint(-random_width, random_width),
                                    curve.evaluate(i * 0.05)[1][0] + random.randint(-random_width, random_width)))
                    x_list.append(curve.evaluate(i * 0.05)[0][0])
                    y_list.append(curve.evaluate(i * 0.05)[1][0])
        mask_img_draw = ImageDraw.Draw(mask_img)
        mask_img_draw.polygon(pt_list, fill=(0, 0, 0))
        mask_tensor = get_tensor(normalize=False, toTensor=True)(mask_img)[0].unsqueeze(0)

        return {"source_img": img_p_np, "reference_img": ref_image, "mask_img": mask_img, "cpation": text,
                "bbox": bbox_pad}