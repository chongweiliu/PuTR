import random
import torch
import cv2
import numpy as np

from math import floor


class MultiCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, infos):
        for t in self.transforms:
            images, infos = t(images, infos)
        return images, infos


class MultiRandomSelect:
    def __init__(self, transform1, transform2, p: float = 0.5):
        self.transform1 = transform1
        self.transform2 = transform2
        self.p = p

    def __call__(self, imgs, infos):
        if random.random() < self.p:
            return self.transform1(imgs, infos)
        return self.transform2(imgs, infos)


class MultiRandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, imgs: list, infos: list):
        if random.random() < self.p:
            def hflip(img, info: dict):
                img = img[:, ::-1]
                h, w = img.shape[:2]
                assert "boxes" in info, "Info do not have key: boxes"
                # x1y1            x1y1    x2y1
                #          =>          =>
                #     x2y2    x2y2            x1y2
                if len(info["boxes"]) > 0:
                    info["boxes"] = (info["boxes"][:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1])
                                     + torch.as_tensor([w, 0, w, 0]))
                return img, info
            for i in range(len(imgs)):
                imgs[i], infos[i] = hflip(imgs[i], infos[i])
        return imgs, infos


class MultiRandomResize:
    """
    随机进行 Resize
    """
    def __init__(self, sizes: list | tuple, max_size=None):
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, imgs, infos):
        new_size = random.choice(self.sizes)

        def resize(img, info, size, max_size):
            def get_new_hw(current_hw, new_short_size, new_long_max_size) -> [int, int]:
                h, w = current_hw
                if new_long_max_size is not None:
                    min_wh, max_wh = float(min(w, h)), float(max(w, h))
                    if max_wh / min_wh * new_short_size > new_long_max_size:
                        new_short_size = int(floor(new_long_max_size * min_wh / max_wh))

                if w < h:
                    new_w = new_short_size
                    new_h = int(round(new_short_size * h / w))
                else:
                    new_h = new_short_size
                    new_w = int(round(new_short_size) * w / h)
                return new_h, new_w

            if isinstance(size, (list, tuple)):
                assert len(size) == 2, f"Size length should be 2, but get {len(size)}."
                new_hw = size[::-1]
            else:
                new_hw = get_new_hw(current_hw=img.shape[:2], new_short_size=size, new_long_max_size=max_size)
            resized_img = cv2.resize(img, new_hw[::-1], interpolation=cv2.INTER_LINEAR)
            ratio_w, ratio_h = (float(s) / float(origin_s) for s, origin_s in zip(resized_img.shape[:2], img.shape[:2]))

            assert "boxes" in info, "Info do not have key: boxes."
            assert "areas" in info, "Info do not have key: areas."
            if len(info["boxes"]) > 0:
                info["boxes"] = info["boxes"] * torch.as_tensor([ratio_w, ratio_h, ratio_w, ratio_h])
                info["areas"] = info["areas"] * ratio_w * ratio_h
            return resized_img, info
        for i in range(len(imgs)):
            imgs[i], infos[i] = resize(imgs[i], infos[i], size=new_size, max_size=self.max_size)
        return imgs, infos

class MultiRandomCrop:
    def __init__(self, min_size: int, max_size: int, overflow_bbox: bool = False):
        self.min_size = min_size
        self.max_size = max_size
        self.overflow_bbox = overflow_bbox

    def __call__(self, imgs, infos):
        img_h, img_w = imgs[0].shape[:2]
        crop_w = random.randint(self.min_size, min(img_w, self.max_size))
        crop_h = random.randint(self.min_size, min(img_h, self.max_size))
        
        if img_h < crop_h or img_w < crop_w:
            raise ValueError(f"Required crop size {(crop_h, crop_w)} is larger than input image size {(img_h, img_w)}")
        
        if img_w == crop_w and img_h == crop_h:
            crop_ijhw = (0, 0, img_h, img_w)
        else:
            i = torch.randint(0, img_h - crop_h + 1, size=(1,)).item()
            j = torch.randint(0, img_w - crop_w + 1, size=(1,)).item()
            crop_ijhw = (i, j, crop_h, crop_w)

        def crop(img, info, ijhw):
            i, j, h, w = ijhw
            cropped_img = img[i:i + h, j:j + w]
            if len(info["boxes"]) > 0:
                info["boxes"] = info["boxes"] - torch.as_tensor([j, i, j, i])
                max_wh = torch.as_tensor([w, h])
                if self.overflow_bbox:  # box overflow is legal
                    boxes = info["boxes"].clone()
                    boxes = torch.min(boxes.reshape(-1, 2, 2), max_wh)
                    boxes = boxes.clamp(min=0)
                    keep_idxs = torch.all(torch.as_tensor(boxes[:, 1, :] > boxes[:, 0, :]), dim=1)
                else:# box overflow is not legal
                    info["boxes"] = torch.min(info["boxes"].reshape(-1, 2, 2), max_wh)
                    info["boxes"] = info["boxes"].clamp(min=0)
                    keep_idxs = torch.all(torch.as_tensor(info["boxes"][:, 1, :] > info["boxes"][:, 0, :]), dim=1)
                    info["boxes"] = info["boxes"].reshape(-1, 4)

                for field in ["labels", "ids", "boxes", "areas"]:
                    info[field] = info[field][keep_idxs]
            return cropped_img, info

        for i in range(len(imgs)):
            imgs[i], infos[i] = crop(imgs[i], infos[i], crop_ijhw)
        return imgs, infos


class MultiHSV:
    """
    From YOLOX [https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/data/data_augment.py] and MOTRv2.
    """
    def __init__(self, hgain=5, sgain=30, vgain=30):
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, imgs, infos):
        hsv_augs = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain]
        hsv_augs *= np.random.randint(0, 2, 3)
        hsv_augs = hsv_augs.astype(np.int16)

        def hsv(img, info):
            img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int16)

            img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
            img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
            img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)

            return cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2RGB), info
        for i in range(len(imgs)):
            imgs[i], infos[i] = hsv(imgs[i], infos[i])
        return imgs, infos
    
    