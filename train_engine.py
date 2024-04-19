# @Author       : Ruopeng Gao
# @Date         : 2022/7/5
import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.distributed
import torch.nn.functional as F
from datetime import datetime
import gc

from typing import List, Tuple, Dict
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from models import build_model, build_position_embedding
from data import build_dataset, build_sampler, build_dataloader
from utils.utils import is_distributed, distributed_rank, set_seed, is_main_process, \
    distributed_world_size
from utils.nested_tensor import tensor_list_to_nested_tensor
from structures.track_instances import TrackInstances
from models.utils import get_model, save_checkpoint, load_checkpoint
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from log.logger import Logger, ProgressLogger
from log.log import MetricLog
from models.utils import load_pretrained_model

torch.multiprocessing.set_sharing_strategy('file_system')


def train(config: dict):
    sub_folder = os.path.join(config["OUTPUTS_DIR"], f"train_{datetime.now().strftime('%Y-%m-%d-%H-%M')}")
    train_logger = Logger(logdir=sub_folder, only_main=True)
    train_logger.show(head="Configs:", log=config)
    train_logger.write(log=config, filename="config.yaml", mode="w")
    train_logger.tb_add_git_version(git_version=config["GIT_VERSION"])

    set_seed(config["SEED"])

    model = build_model(config=config)
    # Load Pretrained Model
    if config["PRETRAINED_MODEL"] is not None:
        model = load_pretrained_model(model, config["PRETRAINED_MODEL"], show_details=False)

    # Data process
    dataset_train = build_dataset(config=config, split="train")
    sampler_train = build_sampler(dataset=dataset_train, shuffle=True)
    dataloader_train = build_dataloader(dataset=dataset_train, sampler=sampler_train,
                                        batch_size=config["BATCH_SIZE"], num_workers=config["NUM_WORKERS"])
    
    # for i, batch in enumerate(dataloader_train):
    #     if i < 10:
    #         img_list = batch["imgs"]
    #         info_list = batch["infos"]
    #         for i, imgs in enumerate(img_list):
    #             for j, img in enumerate(imgs):
    #                 if img.shape[0] == 3:
    #                     img = img.flip(0).permute(1, 2, 0).contiguous().cpu().numpy()
    #                 else:
    #                     img = img[:, :, ::-1]
    #                 if np.min(img) < 0:
    #                     img = (img * 0.2 + 0.5) * 255
    #                 img = img.astype(np.uint8)
    #                 info = info_list[i][j]
    #                 boxes = info["boxes"]
    #                 for box in boxes:
    #                     x1, y1, x2, y2 = box.tolist()
    #                     cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    #                 cv2.imshow("img", img)
    #                 cv2.waitKey(0)
    #                 if "sampled_features" in info:
    #                     sampled_features = info["sampled_features"]
    #                     for sampled_feature in sampled_features:
    #                         sampled_feature = sampled_feature.view(config["PATCH_GRID"], config["PATCH_GRID"], 3)
    #                         sampled_feature = sampled_feature.flip(2).contiguous().cpu().numpy()
    #                         if np.min(sampled_feature) < 0:
    #                             sampled_feature = (sampled_feature * 0.2 + 0.5) * 255
    #                         sampled_feature = sampled_feature.astype(np.uint8)
    #                         cv2.imshow("sampled_feature", sampled_feature)
    #                         cv2.waitKey(0)
    #     else:
    #         break
    #
    # exit(0)


    # Optimizer
    optimizer, lr_names = model.configure_optimizers(config["WEIGHT_DECAY"], config["LR"], torch.device("cuda", distributed_rank()))
    # Scheduler
    if config["LR_SCHEDULER"] == "MultiStep":
        scheduler = MultiStepLR(
            optimizer,
            milestones=config["LR_DROP_MILESTONES"],
            gamma=config["LR_DROP_RATE"]
        )
    elif config["LR_SCHEDULER"] == "Cosine":
        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=config["EPOCHS"]
        )
    else:
        raise ValueError(f"Do not support lr scheduler '{config['LR_SCHEDULER']}'")

    # Training states
    train_states = {
        "start_epoch": 0,
        "global_iters": 0
    }

    # Resume
    if config["RESUME"] is not None:
        if config["RESUME_SCHEDULER"]:
            load_checkpoint(model=model, path=config["RESUME"], states=train_states,
                            optimizer=optimizer, scheduler=scheduler)
        else:
            load_checkpoint(model=model, path=config["RESUME"], states=train_states)
            for _ in range(train_states["start_epoch"]):
                scheduler.step()

    # Set start epoch
    start_epoch = train_states["start_epoch"]

    if is_distributed():
        model = DDP(module=model, device_ids=[distributed_rank()], find_unused_parameters=False)
    
    multi_checkpoint = "MULTI_CHECKPOINT" in config and config["MULTI_CHECKPOINT"]
    
    pe = build_position_embedding(config=config)
    

    # Training:
    for epoch in range(start_epoch, config["EPOCHS"]):
        gc.collect()
        if is_distributed():
            sampler_train.set_epoch(epoch)
        dataset_train.set_epoch(epoch)
        # sampler_train = build_sampler(dataset=dataset_train, shuffle=True)
        # dataloader_train = build_dataloader(
        #     dataset=dataset_train, sampler=sampler_train,
        #     batch_size=config["BATCH_SIZE"], num_workers=config["NUM_WORKERS"])

        lrs = [optimizer.param_groups[_]["lr"] for _ in range(len(optimizer.param_groups))]
        assert len(lrs) == len(lr_names)
        lr_info = [{name: lr} for name, lr in zip(lr_names, lrs)]
        train_logger.show(head=f"[Time={datetime.now().strftime('%Y-%m-%d %H:%M')}, Epoch {epoch}] lr={lr_info}")
        train_logger.write(head=f"[Time={datetime.now().strftime('%Y-%m-%d %H:%M')}, Epoch {epoch}] lr={lr_info}")

        train_logger.tb_add_scalar(tag="lr", scalar_value=lrs[0], global_step=epoch, mode="epochs")

        train_one_epoch(
            config=config,
            pe=pe,
            model=model,
            train_states=train_states,
            max_norm=config["CLIP_MAX_NORM"],
            dataloader=dataloader_train,
            optimizer=optimizer,
            epoch=epoch,
            logger=train_logger,
            accumulation_steps=config["ACCUMULATION_STEPS"],
            multi_checkpoint=multi_checkpoint,
        )
        scheduler.step()
        train_states["start_epoch"] += 1
        if multi_checkpoint is True:
            pass
        else:
            if config["DATASET"] == "DanceTrack" or config["EPOCHS"] < 100 or (epoch + 1) % 5 == 0:
                save_checkpoint(
                    model=model,
                    path=os.path.join(sub_folder, f"checkpoint_{epoch}.pth"),
                    states=train_states,
                    optimizer=optimizer,
                    scheduler=scheduler
                )

    return


def train_one_epoch(config: dict, pe, model, train_states: dict, max_norm: float,
                    dataloader: DataLoader, optimizer: torch.optim,
                    epoch: int, logger: Logger,
                    accumulation_steps: int = 1,
                    multi_checkpoint: bool = False):
    """
    Args:
        model: Model.
        train_states:
        max_norm: clip max norm.
        dataloader: Training dataloader.
        criterion: Loss function.
        optimizer: Training optimizer.
        epoch: Current epoch.
        logger: unified logger.
        accumulation_steps:
        multi_checkpoint:

    Returns:
        Logs
    """
    model.train()
    optimizer.zero_grad()
    device = next(get_model(model).parameters()).device

    dataloader_len = len(dataloader)
    metric_log = MetricLog()
    epoch_start_timestamp = time.time()
    
    
    # 生成采样网格坐标
    n_grid = config["PATCH_GRID"]
    dim = config["DIM"]
    len_sampled_feature = config["PATCH_GRID"] * config["PATCH_GRID"] * 3
    x = torch.linspace(0, n_grid - 1, n_grid, dtype=torch.float32).to(device)
    y = torch.linspace(0, n_grid - 1, n_grid, dtype=torch.float32).to(device)
    yv, xv = torch.meshgrid(x, y)
    
    BECloss = nn.BCELoss()

    iter_start_timestamp = time.time()
    for _i, batch in enumerate(dataloader):
        batch_size = len(batch["infos"])
        n_frames = len(batch["infos"][0])
        n_boxes = torch.zeros(batch_size, n_frames, dtype=torch.int32).to(device)
        sfeatures_list = []
        gts = []
        cxcys = []
        img_hws = []
        _col_gts_coors = []
        id_coors_list = []

        #对物体进行采样生成 token
        for bidx in range(batch_size):
            temp_imgs = batch["imgs"][bidx]
            batch["imgs"][bidx] = None
            temp_imgs = torch.from_numpy(temp_imgs).to(device).float()
            temp_imgs = temp_imgs.permute(0, 3, 1, 2).contiguous().div_(255.).sub_(0.5).div_(0.2)
            temp_sfeatures_list = []
            temp_cxcys = []
            temp_img_hws = []
            id_coors_list.append(batch["infos"][bidx][-1]["id_coors"])
            for fidx in range(n_frames):
                info = batch["infos"][bidx][fidx]# x1y1x2y2
                batch["infos"][bidx][fidx] = None
                box = info["boxes"].to(device)

                # collect gt
                if fidx == n_frames - 1:
                    gts.append(info["gt"])
                    _col_gts_coors.append(info["ids_pervious_coor"])
                img_ = temp_imgs[fidx:fidx+1]# (1, C, H, W)
                img_h, img_w = img_.shape[-2:]
                temp_img_hws.append((img_h, img_w))
                n_boxes[bidx, fidx] = len(box)

                if n_boxes[bidx, fidx] > 0:
                    temp_cxcys.append(((box[:, :2] + box[:, 2:]) / 2))
                    widths = box[:, 2] - box[:, 0] + 0.5
                    heights = box[:, 3] - box[:, 1] + 0.5

                    # 计算每个网格的宽度和高度
                    grid_widths = (widths / n_grid).view(-1, 1, 1)
                    grid_heights = (heights / n_grid).view(-1, 1, 1)

                    # 将网格坐标转化为每个网格的中点坐标
                    grid_x = box[:, 0].view(-1, 1, 1) + (xv + 0.5).unsqueeze(0) * grid_widths
                    grid_y = box[:, 1].view(-1, 1, 1) + (yv + 0.5).unsqueeze(0) * grid_heights

                    # 将每个网格的中点坐标拼接成一个数组
                    grid_coordinates = torch.stack((grid_x, grid_y), dim=3)  # (n, n_grid, n_grid, 2)

                    # 将网格坐标转化为归一化坐标范围(-1, 1)
                    grid_coordinates[:, :, :, 0] = grid_coordinates[:, :, :, 0] / img_w * 2 - 1
                    grid_coordinates[:, :, :, 1] = grid_coordinates[:, :, :, 1] / img_h * 2 - 1

                    # 使用F.grid_sample进行采样
                    sampled_features = F.grid_sample(
                        img_, grid_coordinates.reshape(
                            1, grid_coordinates.shape[0] * grid_coordinates.shape[1], grid_coordinates.shape[2], 2),
                        mode='bilinear',
                        padding_mode='zeros').permute(0, 2, 3, 1).reshape(-1, n_grid * n_grid * img_.shape[1])
                    del grid_coordinates, grid_x, grid_y, grid_widths, grid_heights
                    temp_sfeatures_list.append(sampled_features)

                else:
                    temp_cxcys.append(torch.zeros((0, 2), device=device, dtype=torch.int32))
                    temp_sfeatures_list.append(torch.zeros((0, n_grid * n_grid * img_.shape[1]), device=device))
            sfeatures_list.append(temp_sfeatures_list)
            img_hws.append(temp_img_hws)
            cxcys.append(temp_cxcys)

        # 计算每个batch的最大box数
        max_nbox_per_frame, _ = n_boxes.max(axis=0)
        start_coor = torch.cumsum(max_nbox_per_frame, dim=0)
        start_coor = torch.cat([torch.tensor([0]).to(device), start_coor]) + 1
        start_coor = start_coor.tolist()
        #生成每个 id 在每帧中的绝对坐标
        for bidx in range(batch_size):
            id_coors = id_coors_list[bidx]
            for id in id_coors.keys():
                id_coors[id] = [start_coor[coor[0]] + coor[1] for coor in id_coors[id]]

        # 生成gt的坐标
        col_gts_coors = []
        for coors in _col_gts_coors:
            temp_list = []
            for coor in coors:
                if coor[0] != -1:
                    temp_list.append(start_coor[coor[0]] + coor[1].item())
            col_gts_coors.append(temp_list)
        #生成输入 token 列表以及对应掩码
        inputs = torch.zeros(batch_size, 1 + max_nbox_per_frame.sum(), len_sampled_feature, device=device)
        inputs_pos = torch.zeros(batch_size, 1 + max_nbox_per_frame.sum(), dim, device=device)
        inputs_mask = torch.zeros(batch_size, 1 + max_nbox_per_frame.sum(), dtype=torch.float32, device=device)

        for fidx in range(n_frames):
            for bidx in range(batch_size):
                inputs[bidx, start_coor[fidx]:start_coor[fidx] + n_boxes[bidx, fidx]] = sfeatures_list[bidx][fidx]
                inputs_mask[bidx, start_coor[fidx]:start_coor[fidx] + n_boxes[bidx, fidx]] = 1
                cxcy = cxcys[bidx][fidx]
                img_h, img_w = img_hws[bidx][fidx]
                if cxcy.shape[0] > 0:
                    inputs_pos[bidx, start_coor[fidx]:start_coor[fidx] + n_boxes[bidx, fidx]] = pe(cxcy[:, 0], cxcy[:, 1], img_h, img_w)


        # attention mask
        atten_mask = (inputs_mask.unsqueeze(dim=2) @ inputs_mask.unsqueeze(dim=1)).bool()
        link_mask = torch.zeros_like(atten_mask)

        for bidx in range(batch_size):
            id_coors = id_coors_list[bidx]
            for coors in id_coors.values():
                coors = torch.as_tensor(coors, device=device, dtype=torch.int32)
                xc, yc = torch.meshgrid(coors, coors)
                link_mask[bidx].index_put_((xc.flatten(), yc.flatten()), torch.tensor(1, device=device, dtype=torch.bool))
            link_mask[bidx, start_coor[-2]:] = 0
            link_mask[bidx, start_coor[-2]:, col_gts_coors[bidx]] = 1

        for fidx in range(n_frames):
            atten_mask[:, start_coor[fidx]:start_coor[fidx + 1], start_coor[fidx]:start_coor[fidx + 1]] = 0
        atten_mask *= link_mask
        atten_mask.tril_(diagonal=0)

        atten_mask[:, torch.arange(atten_mask.shape[1]), torch.arange(atten_mask.shape[1])] = 1
        atten_mask[:, :, 0] = 1

        atten_mask = atten_mask.unsqueeze(dim=1)

        h = model(inputs, inputs_pos, atten_mask)

        loss = 0
        for bidx in range(batch_size):
            gt = gts[bidx].to(device)
            if gt.shape[0] == 0:
                continue
            temp = h[bidx, col_gts_coors[bidx]] @ h[bidx, start_coor[-2]:start_coor[-2] + gt.shape[1]].T
            loss += BECloss(temp.sigmoid(), gt)
        loss /= batch_size

        # Metrics log
        metric_log.update(name="loss", value=loss.item())
        loss = loss / accumulation_steps
        loss.backward()

        if (_i + 1) % accumulation_steps == 0:
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            else:
                pass
            optimizer.step()
            optimizer.zero_grad()

        del atten_mask, link_mask, inputs, inputs_pos, inputs_mask, h, loss



        # For logging
        iter_end_timestamp = time.time()
        metric_log.update(name="time per iter", value=iter_end_timestamp-iter_start_timestamp)
        iter_start_timestamp = iter_end_timestamp
        # Outputs logs
        if _i % 10 == 0:
            metric_log.sync()
            max_memory = max([torch.cuda.max_memory_allocated(torch.device('cuda', _i))
                              for _i in range(distributed_world_size())]) // (1024**2)
            second_per_iter = metric_log.metrics["time per iter"].avg
            logger.show(head=f"[Time={datetime.now().strftime('%Y-%m-%d %H:%M')}, Epoch={epoch}, Iter={_i}, "
                             f"{second_per_iter:.2f}s/iter, "
                             f"{_i}/{dataloader_len} iters, "
                             f"rest time: {int(second_per_iter * (dataloader_len - _i) // 60)} min, "
                             f"Max Memory={max_memory}MB]",
                        log=metric_log)
            logger.write(head=f"[Time={datetime.now().strftime('%Y-%m-%d %H:%M')}, Epoch={epoch}, Iter={_i}/{dataloader_len}]",
                         log=metric_log, filename="log.txt", mode="a")
            logger.tb_add_metric_log(log=metric_log, steps=train_states["global_iters"], mode="iters")

        if multi_checkpoint:
            if _i % 100 == 0 and is_main_process():
                save_checkpoint(
                    model=model,
                    path=os.path.join(logger.logdir[:-5], f"checkpoint_{int(_i // 100)}.pth")
                )

        train_states["global_iters"] += 1

    # Epoch end
    metric_log.sync()
    epoch_end_timestamp = time.time()
    epoch_minutes = int((epoch_end_timestamp - epoch_start_timestamp) // 60)
    logger.show(head=f"[Time={datetime.now().strftime('%Y-%m-%d %H:%M')}, Epoch: {epoch}, Total Time: {epoch_minutes}min]",
                log=metric_log)
    logger.write(head=f"[Time={datetime.now().strftime('%Y-%m-%d %H:%M')}, Epoch: {epoch}, Total Time: {epoch_minutes}min]",
                 log=metric_log, filename="log.txt", mode="a")
    logger.tb_add_metric_log(log=metric_log, steps=epoch, mode="epochs")

    return


# class BCEFocalLoss(torch.nn.Module):
#     """
#     二分类的Focalloss alpha 固定
#     """
#
#     def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
#         super().__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.reduction = reduction
#
#     def forward(self, _input, target):
#         pt = torch.sigmoid(_input)
#         alpha = self.alpha
#         loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
#                (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
#         if self.reduction == 'elementwise_mean':
#             return torch.mean(loss)
#         elif self.reduction == 'sum':
#             return torch.sum(loss)