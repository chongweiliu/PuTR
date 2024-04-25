# Copyright (c) Ruopeng Gao. All Rights Reserved.
import os
import cv2

import torchvision.transforms.functional as F

from torch.utils.data import Dataset
from math import floor
import numpy as np
class SeqDataset(Dataset):
    def __init__(self, seq_dir: str):
        # a hack implementation for BDD100K and others:
        if "BDD100K" in seq_dir:
            image_paths = sorted(os.listdir(os.path.join(seq_dir)))
            image_paths = [os.path.join(seq_dir, _) for _ in image_paths if ("jpg" in _) or ("png" in _)]
        else:
            image_paths = sorted(os.listdir(os.path.join(seq_dir, "img1")))
            image_paths = [os.path.join(seq_dir, "img1", _) for _ in image_paths if ("jpg" in _) or ("png" in _)]
        
        if 'DanceTrack' in seq_dir:
            det_path = os.path.join(seq_dir, 'det', 'byte065.txt')
        elif 'SportsMOT' in seq_dir:
            det_path = os.path.join(seq_dir, 'det', 'yolox_x_train.txt')
            # det_path = os.path.join(seq_dir, 'det', 'yolox_x_mix.txt')
        elif 'MOT17' in seq_dir:
            det_path = os.path.join(seq_dir, 'det', 'byte065.txt')
        elif 'MOT20' in seq_dir:
            det_path = os.path.join(seq_dir, 'det', 'byte065.txt')
        else:
            raise NotImplementedError(f"SeqDataset DO NOT support dataset '{seq_dir}'")
        self.seq_dir = seq_dir
        self.image_paths = image_paths
        self.image_height = 800
        self.image_width = 1536
        self.det_path = det_path
        self.dets = self.load_dets()
    
        return

    @staticmethod
    def load(path):
        image = cv2.imread(path)
        assert image is not None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def process_image(self, image, item):
        orig_h, orig_w = image.shape[:2]
        new_short_size = self.image_height
        if self.image_width is not None:
            min_wh, max_wh = float(min(orig_w, orig_h)), float(max(orig_w, orig_h))
            if max_wh / min_wh * self.image_height > self.image_width:
                new_short_size = int(floor(self.image_width * min_wh / max_wh))

        if orig_w < orig_h:
            new_w = new_short_size
            new_h = int(round(new_short_size * orig_h / orig_w))
        else:
            new_h = new_short_size
            new_w = int(round(new_short_size) * orig_w / orig_h)

        image = cv2.resize(image, (new_w, new_h))
        
        ratio_w, ratio_h = new_w / orig_w, new_h / orig_h
        
        orig_dets = self.dets[self.dets[:, 0] == item + 1][:, 1:]
        
        dets = orig_dets.copy()
        
        idxs_det2odet = None
        if len(dets) > 0:
            scores = dets[:, -1:]
            dets = dets[:, :-1]
            dets = dets * np.array([ratio_w, ratio_h, ratio_w, ratio_h], dtype=np.float32)
            # box overflow is not legal
            dets = dets.clip(min=0)
            dets = dets.clip(max=[new_w, new_h, new_w, new_h])
            keep_idxs = np.all(dets[:, 2:] > dets[:, :2], axis=1)
            dets = np.concatenate([dets, scores], axis=1)
            dets = dets[keep_idxs]
            orig_dets = orig_dets[keep_idxs]
            
            if 'DanceTrack' in self.seq_dir:
                dets = np.concatenate([dets, np.ones((len(dets), 1), dtype=np.float32)], axis=1)
                orig_dets = np.concatenate([orig_dets, np.ones((len(orig_dets), 1), dtype=np.float32)], axis=1)
            elif 'SportsMOT' in self.seq_dir:
                dets = np.concatenate([dets, np.ones((len(dets), 1), dtype=np.float32)], axis=1)
                orig_dets = np.concatenate([orig_dets, np.ones((len(orig_dets), 1), dtype=np.float32)], axis=1)
            elif 'MOT17' in self.seq_dir:
                dets = np.concatenate([dets, np.ones((len(dets), 1), dtype=np.float32)], axis=1)
                orig_dets = np.concatenate([orig_dets, np.ones((len(orig_dets), 1), dtype=np.float32)], axis=1)
            elif 'MOT20' in self.seq_dir:
                dets = np.concatenate([dets, np.ones((len(dets), 1), dtype=np.float32)], axis=1)
                orig_dets = np.concatenate([orig_dets, np.ones((len(orig_dets), 1), dtype=np.float32)], axis=1)
            else:
                raise NotImplementedError(f"SeqDataset DO NOT support dataset '{self.seq_dir}'")
        
        return image, dets.astype(np.float32), orig_dets.astype(np.float32)
    def load_dets(self):
        dets = []
        with open(self.det_path, 'r') as f:
            for line in f:
                # gt per line: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, 1, 1, 1
                # https://github.com/DanceTrack/DanceTrack
                t, i, *xywh, s = line.strip().split(",")[:7]
                t, s = map(float, [t, s])
                x, y, w, h = map(float, xywh)
                dets.append([t, x, y, x + w, y + h, s]) # [t, x1, y1, x2, y2]
            
        return np.array(dets, dtype=np.float32)
    
    
    
    def __getitem__(self, item):
        image = self.load(self.image_paths[item])
        return self.process_image(image=image, item=item)

    def __len__(self):
        return len(self.image_paths)
