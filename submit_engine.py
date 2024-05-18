import os
import json
import torch
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
from os import path
from torch.utils.data import DataLoader
import cv2
import shutil
import numpy as np
import os
import time


from models import build_model
from models.utils import load_checkpoint
from log.logger import Logger
from data.seq_dataset import SeqDataset
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
        new_dict = {}
        
        time_per_frame = []
        for i, (img, dets, orig_dets) in enumerate(tqdm(self.dataloader, desc=f"Submit seq: {self.seq_name}")):
            start_time = time.time()
            
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
            
            end_time = time.time()
            time_per_frame.append(end_time - start_time)
            
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
                        if track_id in new_dict:
                            track_dict[track_id].extend(new_dict[track_id])
                            new_dict.pop(track_id)
                        if track_id in track_hit_dict:
                            track_dict[track_id].extend(track_hit_dict[track_id])
                            track_hit_dict.pop(track_id)
                elif state == TS.New:
                    new_dict.setdefault(track_id,[]).append([frame_id, track_id, *tlwh])

        self.write_results(tracks_result=track_dict)
        return np.nanmean(time_per_frame)

    def write_results(self, tracks_result: dict):
        assert self.dataset_name in ["DanceTrack", "SportsMOT", "MOT17", "MOT20"], f"{self.dataset_name} dataset is not supported for submit process."
        
        with open(os.path.join(self.predict_dir, f"{self.seq_name}.txt"), "a") as file:
            for value in tracks_result.values():
                for line in value:
                    result_line = f"{int(line[0])}," \
                                  f"{int(line[1])}," \
                                  f"{line[2]},{line[3]},{line[4]},{line[5]},1,-1,-1,-1\n"
                    file.write(result_line)
        print(f"Write results to {self.predict_dir}/{self.seq_name}.txt")
        if self.dataset_name in ["MOT17"]:
            shutil.copyfile(os.path.join(self.predict_dir, f"{self.seq_name}.txt"),
                            os.path.join(self.predict_dir, f"{self.seq_name.replace('FRCNN', 'DPM')}.txt"))
            shutil.copyfile(os.path.join(self.predict_dir, f"{self.seq_name}.txt"),
                            os.path.join(self.predict_dir, f"{self.seq_name.replace('FRCNN', 'SDP')}.txt"))
            print(f"Copy results to {self.predict_dir}/{self.seq_name.replace('FRCNN', 'DPM')}.txt")
            print(f"Copy results to {self.predict_dir}/{self.seq_name.replace('FRCNN', 'SDP')}.txt")
            
            
        if self.dataset_name in ["MOT17", "MOT20"]:
            print(f"Run interpolation for {self.seq_name}")
            n_dti = 20 if self.dataset_name == "MOT20" else 3
            tracks_result = dti(f"{self.predict_dir}/{self.seq_name}.txt", n_dti=n_dti)
            if not os.path.exists(self.predict_dir + f"/dti"):
                os.makedirs(self.predict_dir + f"/dti")
            with open(self.predict_dir + f"/dti/{self.seq_name}.txt", "a") as file:
                for line in tracks_result:
                    result_line = f"{int(line[0])}," \
                                  f"{int(line[1])}," \
                                  f"{line[2]},{line[3]},{line[4]},{line[5]},1,-1,-1,-1\n"
                    file.write(result_line)
            print(f"Write interpolated results to {self.predict_dir}/dti/{self.seq_name}.txt")
            if self.dataset_name in ["MOT17"]:
                shutil.copyfile(self.predict_dir + f"/dti/{self.seq_name}.txt",
                                self.predict_dir + f"/dti/{self.seq_name.replace('FRCNN', 'DPM')}.txt")
                shutil.copyfile(self.predict_dir + f"/dti/{self.seq_name}.txt",
                                self.predict_dir + f"/dti/{self.seq_name.replace('FRCNN', 'SDP')}.txt")
                print(f"Copy interpolated results to {self.predict_dir}/dti/{self.seq_name.replace('FRCNN', 'DPM')}.txt")
                print(f"Copy interpolated results to {self.predict_dir}/dti/{self.seq_name.replace('FRCNN', 'SDP')}.txt")
            
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
        seq_names = os.listdir(data_split_dir)
    elif dataset_name == "MOT17":
        data_split_dir = path.join(data_root, dataset_name, dataset_split)
        seq_names = os.listdir(data_split_dir)
        seq_names = [seq_name for seq_name in seq_names if "FRCNN" in seq_name]
    elif dataset_name == "MOT20":
        data_split_dir = path.join(data_root, dataset_name, dataset_split)
        seq_names = os.listdir(data_split_dir)
    else:
        raise NotImplementedError(f"Do not support this Dataset name: {dataset_name}")
    
    fps_list = []
    n_seq = len(seq_names)
    for idx, seq_name in enumerate(seq_names):
        print(f"{idx + 1}/{n_seq}: {seq_name}")
        
        seq_name = str(seq_name)
        submitter = Submitter(
            config=config,
            dataset_name=dataset_name,
            split_dir=data_split_dir,
            seq_name=seq_name,
            outputs_dir=outputs_dir,
            model=model
        )
        avg_time_per_frame = submitter.run()
        fps = 1. / avg_time_per_frame
        print(f"{seq_name} FPS: {fps}")
        fps_list.append(fps)
    print(f"Average FPS: {np.nanmean(fps_list)}")
    
    if dataset_split == "val":
        tracker_dir = os.path.join(outputs_dir, "tracker")
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

def dti(txt_path, n_min=25, n_dti=20): #n_dti MOT20 20; MOT17 3
    seq_data = np.loadtxt(txt_path, dtype=np.float64, delimiter=',')
    min_id = int(np.min(seq_data[:, 1]))
    max_id = int(np.max(seq_data[:, 1]))
    seq_results = np.zeros((1, 10), dtype=np.float64)
    for track_id in range(min_id, max_id + 1):
        index = (seq_data[:, 1] == track_id)
        tracklet = seq_data[index]
        # sort
        tracklet = tracklet[tracklet[:, 0].argsort()]
        tracklet_dti = tracklet
        if tracklet.shape[0] == 0:
            continue
        n_frame = tracklet.shape[0]
        if n_frame > n_min:
            frames = tracklet[:, 0]
            frames_dti = {}
            for i in range(0, n_frame):
                right_frame = frames[i]
                if i > 0:
                    left_frame = frames[i - 1]
                else:
                    left_frame = frames[i]
                # disconnected track interpolation
                if 1 < right_frame - left_frame < n_dti:
                    num_bi = int(right_frame - left_frame - 1)
                    right_bbox = tracklet[i, 2:6]
                    left_bbox = tracklet[i - 1, 2:6]
                    for j in range(1, num_bi + 1):
                        curr_frame = j + left_frame
                        curr_bbox = (curr_frame - left_frame) * (right_bbox - left_bbox) / \
                                    (right_frame - left_frame) + left_bbox
                        frames_dti[curr_frame] = curr_bbox
            num_dti = len(frames_dti.keys())
            if num_dti > 0:
                data_dti = np.zeros((num_dti, 10), dtype=np.float64)
                for n in range(num_dti):
                    data_dti[n, 0] = list(frames_dti.keys())[n]
                    data_dti[n, 1] = track_id
                    data_dti[n, 2:6] = frames_dti[list(frames_dti.keys())[n]]
                    data_dti[n, 6:] = [1, -1, -1, -1]
                tracklet_dti = np.vstack((tracklet, data_dti))
        seq_results = np.vstack((seq_results, tracklet_dti))
    seq_results = seq_results[1:]
    seq_results = seq_results[seq_results[:, 0].argsort()]
    return seq_results