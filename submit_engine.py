# Copyright (c) Ruopeng Gao. All Rights Reserved.
import os
import json
import torch
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
from os import path
from typing import List
from torch.utils.data import DataLoader
import cv2

from models import build_model
from models.utils import load_checkpoint, get_model
from models.runtime_tracker import RuntimeTracker
from utils.utils import yaml_to_dict, is_distributed, distributed_world_size, distributed_rank, inverse_sigmoid
from utils.nested_tensor import tensor_list_to_nested_tensor
from utils.box_ops import box_cxcywh_to_xyxy
from log.logger import Logger
from data.seq_dataset import SeqDataset
from structures.track_instances import TrackInstances
from models.runtime_tracker import RuntimeTracker, TI, TS
import random

random.seed(0)
def generate_random_colors(num_colors):
    # Generate a list of random colors
    color_list = []
    for i in range(num_colors):
        color_tuple = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        color_list.append(color_tuple)

    return color_list


class Submitter:
    def __init__(self, config, dataset_name: str, split_dir: str, seq_name: str, outputs_dir: str, model: nn.Module):
        self.dataset_name = dataset_name
        self.seq_name = seq_name
        self.seq_dir = path.join(split_dir, seq_name)
        self.outputs_dir = outputs_dir
        self.predict_dir = path.join(self.outputs_dir, "tracker")
        self.model = model
        
        self.tracker = RuntimeTracker(config, model)

        self.dataset = SeqDataset(seq_dir=self.seq_dir)
        self.dataloader = DataLoader(self.dataset, batch_size=1, num_workers=4, shuffle=False)
        self.device = next(self.model.parameters()).device

        self.visualize = config["VISUALIZE"]
        # 对路径进行一些操作
        os.makedirs(self.predict_dir, exist_ok=True)
        if os.path.exists(os.path.join(self.predict_dir, f'{self.seq_name}.txt')):
            os.remove(os.path.join(self.predict_dir, f'{self.seq_name}.txt'))
        self.model.eval()
        
        self.random_color_list = generate_random_colors(100)
        
        self.min_track_hits = config["MIN_TRACK_HITS"]
        
        
        
        
        
        return

    @torch.no_grad()
    def run(self):
        track_hit_dict = {}
        track_dict = {}
        lost_dict = {}
        new_dict = {}
        
        bdd100k_results = []    # for bdd100k, will be converted into json file, different from other datasets.
        for i, (img, dets, orig_dets) in enumerate(tqdm(self.dataloader, desc=f"Submit seq: {self.seq_name}")):
            # if self.seq_name != "dancetrack0004":
            #     continue
            # img = img[0].cpu().numpy()[:, :, ::-1].astype("uint8")
            # dets = dets[0].cpu().numpy().astype("int32")
            # for j in range(len(dets)):
            #     cv2.rectangle(img, (dets[j][0], dets[j][1]), (dets[j][2], dets[j][3]), (0, 255, 0), 2)
            # cv2.imshow("img", img)
            # cv2.waitKey(1)
            
            orig_img = img[0].clone()
            orig_img = orig_img.cpu().numpy()[:, :, ::-1].astype("uint8")

            img, dets, orig_dets = img.to(self.device), dets.to(self.device), orig_dets.to(self.device)
            dets, orig_dets = dets[0], orig_dets[0]

            trks = self.tracker.update(img, dets, orig_dets)
            trks = trks.cpu().numpy()

            dets = trks[:, TI.TLBR:TI.TLBR + 4].astype("int32")
            orig_dets = trks[:, TI.ORIG_TLBR:TI.ORIG_TLBR + 4].astype("float32")
            trk_cls = trks[:, TI.ClassID].astype("int32")
            trk_scores = trks[:, TI.Score].astype("float32")
            trk_states = trks[:, TI.State].astype("int32")
            trk_ids = trks[:, TI.TrackID].astype("int32")
            trk_hits = trks[:, TI.Hits].astype("int32")

            if self.visualize:
                for xyxy, orig_xyxy, cls, score, state, idx in zip(dets, orig_dets, trk_cls, trk_scores, trk_states, trk_ids):
                    if state == TS.New:
                        cv2.rectangle(orig_img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 255, 255), 2)
                    elif state == TS.Track:
                        cv2.rectangle(orig_img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), self.random_color_list[idx % 100], 2)
                    elif state == TS.Lost:
                        cv2.rectangle(orig_img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 125, 255), 2)
                    cv2.putText(orig_img, f"{idx}", (xyxy[0], xyxy[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(orig_img, f"Frame: {i}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.imshow("img", orig_img)
                cv2.waitKey(1)

            for idx in range(len(dets)):
                xyxy = orig_dets[idx]
                tlwh = [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]
                track_id = trk_ids[idx]
                frame_id = i + 1
                state = trk_states[idx]
                hit = trk_hits[idx]
                if state == TS.Track:
                    if hit <= 3:
                        track_hit_dict.setdefault(track_id,[]).append([frame_id, track_id, *tlwh])
                    else:
                        track_dict.setdefault(track_id,[]).append([frame_id, track_id, *tlwh])

                        if self.dataset_name in ["MOT17", "MOT20"]:
                            if track_id in lost_dict:
                                track_dict[track_id].extend(lost_dict[track_id])
                                lost_dict.pop(track_id)
                        if track_id in new_dict:
                            track_dict[track_id].extend(new_dict[track_id])
                            new_dict.pop(track_id)
                        if track_id in track_hit_dict:
                            track_dict[track_id].extend(track_hit_dict[track_id])
                            track_hit_dict.pop(track_id)
                elif state == TS.Lost:
                    lost_dict.setdefault(track_id,[]).append([frame_id, track_id, *tlwh])
                elif state == TS.New:
                    new_dict.setdefault(track_id,[]).append([frame_id, track_id, *tlwh])

        if self.dataset_name == "BDD100K":
            raise NotImplementedError("BDD100K dataset is not supported for submit process.")
            # self.update_results(tracks_result=tracks_result, frame_idx=i, results=bdd100k_results, img_path=info[0])
        else:
            self.write_results(tracks_result=track_dict)


        if self.dataset_name == "BDD100K":
            with open(os.path.join(self.predict_dir, '{}.json'.format(self.seq_name)), 'w', encoding='utf-8') as f:
                json.dump(bdd100k_results, f)

        return

    @staticmethod
    def filter_by_score(tracks: TrackInstances, thresh: float = 0.7):
        keep = torch.max(tracks.scores, dim=-1).values > thresh
        return tracks[keep]

    @staticmethod
    def filter_by_area(tracks: TrackInstances, thresh: int = 100):
        assert len(tracks.area) == len(tracks.ids), f"Tracks' 'area' should have the same dim with 'ids'"
        keep = tracks.area > thresh
        return tracks[keep]

    def update_results(self, tracks_result: TrackInstances, frame_idx: int, results: list, img_path: str):
        # Only be used for BDD100K:
        bdd_cls2label = {
            1: "pedestrian",
            2: "rider",
            3: "car",
            4: "truck",
            5: "bus",
            6: "train",
            7: "motorcycle",
            8: "bicycle"
        }
        frame_result = {
            "name": img_path.split("/")[-1],
            "videoName": img_path.split("/")[-1][:-12],
            # "frameIndex": int(img_path.split("/")[-1][:-4].split("-")[-1]) - 1
            "frameIndex": frame_idx,
            "labels": []
        }
        for i in range(len(tracks_result)):
            x1, y1, x2, y2 = tracks_result.boxes[i].tolist()
            ID = str(tracks_result.ids[i].item())
            label = bdd_cls2label[tracks_result.labels[i].item() + 1]
            frame_result["labels"].append(
                {
                    "id": ID,
                    "category": label,
                    "box2d": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    }
                }
            )
        results.append(frame_result)
        return

    def write_results(self, tracks_result: dict):
        assert self.dataset_name in ["DanceTrack", "SportsMOT", "MOT17", "MOT17_SPLIT"], f"{self.dataset_name} dataset is not supported for submit process."
        
        with open(os.path.join(self.predict_dir, f"{self.seq_name}.txt"), "a") as file:
            for value in tracks_result.values():
                for line in value:
                    result_line = f"{line[0]}," \
                                  f"{line[1]}," \
                                  f"{line[2]},{line[3]},{line[4]},{line[5]},1,-1,-1,-1\n"
                    file.write(result_line)
        print(f"Write results to {self.predict_dir}/{self.seq_name}.txt")
        return


def submit(config: dict):
    assert config["OUTPUTS_DIR"] is not None, f"'--outputs-dir' must not be None for submit process."
    assert config["SUBMIT_MODEL"] is not None, f"'--submit-model' must not be None for submit process."
    assert config["SUBMIT_DATA_SPLIT"] is not None, f"'--submit-data-split' must not be None for submit process."
    outputs_dir = os.path.join(
            config["OUTPUTS_DIR"], f"{config['SUBMIT_DATA_SPLIT']}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")
    submit_logger = Logger(
        logdir=outputs_dir,
        only_main=True)
    submit_logger.show(head="Configs:", log=config)
    submit_logger.write(log=config, filename="config.yaml", mode="w")
    os.mkdir(outputs_dir + "/tracker")
    
    data_root = config["DATA_ROOT"]
    dataset_name = config["DATASET"]
    dataset_split = config["SUBMIT_DATA_SPLIT"]

    model = build_model(config=config)
    load_checkpoint(
        model=model,
        path=config["SUBMIT_MODEL"]
    )
    if dataset_name == "DanceTrack" or dataset_name == "SportsMOT":
        data_split_dir = path.join(data_root, dataset_name, dataset_split)
    elif dataset_name == "BDD100K":
        data_split_dir = path.join(data_root, dataset_name, "images/track/", dataset_split)
    else:
        data_split_dir = path.join(data_root, dataset_name, "images", dataset_split)
    seq_names = os.listdir(data_split_dir)

    for seq_name in seq_names:
        seq_name = str(seq_name)
        submitter = Submitter(
            config=config,
            dataset_name=dataset_name,
            split_dir=data_split_dir,
            seq_name=seq_name,
            outputs_dir=outputs_dir,
            model=model
        )
        submitter.run()
    
    if dataset_split == "val":
        tracker_dir = os.path.join(outputs_dir, "tracker")
        # 进行指标计算
        data_dir = os.path.join(data_root, dataset_name)
        if dataset_name == "DanceTrack" or dataset_name == "SportsMOT":
            gt_dir = os.path.join(data_dir, dataset_split)
        elif "MOT17" in dataset_name:
            gt_dir = os.path.join(data_dir, "images", dataset_split)
        else:
            raise NotImplementedError(f"Eval Engine DO NOT support dataset '{dataset_name}'")
        if dataset_name == "DanceTrack" or dataset_name == "SportsMOT":
            os.system(
                f"python3 TrackEval/scripts/run_mot_challenge.py --SPLIT_TO_EVAL {dataset_split}  "
                f"--METRICS HOTA CLEAR Identity  --GT_FOLDER {gt_dir} "
                f"--SEQMAP_FILE {os.path.join(data_dir, f'{dataset_split}_seqmap.txt')} "
                f"--SKIP_SPLIT_FOL True --TRACKERS_TO_EVAL '' --TRACKER_SUB_FOLDER ''  --USE_PARALLEL True "
                f"--NUM_PARALLEL_CORES 8 --PLOT_CURVES False "
                f"--TRACKERS_FOLDER {tracker_dir}")
        elif "MOT17" in dataset_name:
            if "mot15" in dataset_split:
                os.system(
                    f"python3 TrackEval/scripts/run_mot_challenge.py --SPLIT_TO_EVAL {dataset_split}  "
                    f"--METRICS HOTA CLEAR Identity  --GT_FOLDER {gt_dir} "
                    f"--SEQMAP_FILE {os.path.join(data_dir, f'{dataset_split}_seqmap.txt')} "
                    f"--SKIP_SPLIT_FOL True --TRACKERS_TO_EVAL '' --TRACKER_SUB_FOLDER ''  --USE_PARALLEL True "
                    f"--NUM_PARALLEL_CORES 8 --PLOT_CURVES False "
                    f"--TRACKERS_FOLDER {tracker_dir} --BENCHMARK MOT15")
            else:
                os.system(
                    f"python3 TrackEval/scripts/run_mot_challenge.py --SPLIT_TO_EVAL {dataset_split}  "
                    f"--METRICS HOTA CLEAR Identity  --GT_FOLDER {gt_dir} "
                    f"--SEQMAP_FILE {os.path.join(data_dir, f'{dataset_split}_seqmap.txt')} "
                    f"--SKIP_SPLIT_FOL True --TRACKERS_TO_EVAL '' --TRACKER_SUB_FOLDER ''  --USE_PARALLEL True "
                    f"--NUM_PARALLEL_CORES 8 --PLOT_CURVES False "
                    f"--TRACKERS_FOLDER {tracker_dir} --BENCHMARK MOT17")
        else:
            raise NotImplementedError(f"Do not support this Dataset name: {dataset_name}")
        
        metric_path = os.path.join(tracker_dir, "pedestrian_summary.txt")
        with open(metric_path) as f:
            metric_names = f.readline()[:-1].split(" ")
            metric_values = f.readline()[:-1].split(" ")
        metrics = {
            n: float(v) for n, v in zip(metric_names, metric_values)
        }
    return
