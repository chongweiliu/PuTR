import os
import time
import torch
import torch.nn as nn
import torch.distributed
import torch.nn.functional as F
from datetime import datetime
import gc

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from models import build_model, build_xy_pe, build_frame_pe
from data import build_dataset, build_sampler, build_dataloader
from utils.utils import is_distributed, distributed_rank, set_seed, is_main_process, \
    distributed_world_size
from models.utils import get_model, save_checkpoint, load_checkpoint
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from log.logger import Logger
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
    
    xy_pe = build_xy_pe(config)
    frame_pe = build_frame_pe(config)
    

    # Training:
    for epoch in range(start_epoch, config["EPOCHS"]):
        gc.collect()
        if is_distributed():
            sampler_train.set_epoch(epoch)
        dataset_train.set_epoch(epoch)

        lrs = [optimizer.param_groups[_]["lr"] for _ in range(len(optimizer.param_groups))]
        assert len(lrs) == len(lr_names)
        lr_info = [{name: lr} for name, lr in zip(lr_names, lrs)]
        train_logger.show(head=f"[Time={datetime.now().strftime('%Y-%m-%d %H:%M')}, Epoch {epoch}] lr={lr_info}")
        train_logger.write(head=f"[Time={datetime.now().strftime('%Y-%m-%d %H:%M')}, Epoch {epoch}] lr={lr_info}")

        train_logger.tb_add_scalar(tag="lr", scalar_value=lrs[0], global_step=epoch, mode="epochs")

        train_one_epoch(
            config=config,
            xy_pe=xy_pe,
            frame_pe=frame_pe,
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


def train_one_epoch(config: dict, xy_pe, frame_pe, model, train_states: dict, max_norm: float,
                    dataloader: DataLoader, optimizer: torch.optim,
                    epoch: int, logger: Logger,
                    accumulation_steps: int = 1,
                    multi_checkpoint: bool = False):

    model.train()
    optimizer.zero_grad()
    device = next(get_model(model).parameters()).device

    dataloader_len = len(dataloader)
    metric_log = MetricLog()
    epoch_start_timestamp = time.time()
    
    n_grid = config["PATCH_GRID"]
    dim = config["DIM"]
    len_sampled_feature = config["PATCH_GRID"] * config["PATCH_GRID"] * 3
    x = torch.linspace(0, n_grid - 1, n_grid, dtype=torch.float32).to(device)
    y = torch.linspace(0, n_grid - 1, n_grid, dtype=torch.float32).to(device)
    yv, xv = torch.meshgrid(x, y)
    
    CEloss = nn.CrossEntropyLoss()
    iter_start_timestamp = time.time()
    for _i, batch in enumerate(dataloader):
        batch_size = len(batch["infos"])
        n_frames = len(batch["infos"][0])
        n_boxes = torch.zeros(batch_size, n_frames, dtype=torch.int32).to(device)
        sfeatures_list = []
        gts = []
        cxcys = []
        img_hws = []
        _col_gts_coords = []
        id_coords_list = []
        n_ids = []

        for bidx in range(batch_size):
            temp_imgs = batch["imgs"][bidx]
            batch["imgs"][bidx] = None
            temp_imgs = torch.from_numpy(temp_imgs).to(device).float()
            temp_imgs = temp_imgs.permute(0, 3, 1, 2).contiguous().div_(255.).sub_(0.5).div_(0.2)
            temp_sfeatures_list = []
            temp_cxcys = []
            temp_img_hws = []
            temp_gts = []
            temp__col_gts_coords = []
            
            id_coords_list.append(batch["infos"][bidx][-1]["id_coords"])
            n_ids.append(len(id_coords_list[-1]))
            
            for fidx in range(n_frames):
                info = batch["infos"][bidx][fidx]# x1y1x2y2
                batch["infos"][bidx][fidx] = None
                box = info["boxes"].to(device)

                # collect gt
                if fidx > 0:
                    temp_gts.append(info["gt"])
                    temp__col_gts_coords.append(info["ids_pervious_coord"])
                img_ = temp_imgs[fidx:fidx+1]# (1, C, H, W)
                img_h, img_w = img_.shape[-2:]
                temp_img_hws.append((img_h, img_w))
                n_boxes[bidx, fidx] = len(box)

                if n_boxes[bidx, fidx] > 0:
                    temp_cxcys.append(((box[:, :2] + box[:, 2:]) / 2))
                    widths = box[:, 2] - box[:, 0] + 0.5
                    heights = box[:, 3] - box[:, 1] + 0.5

                    grid_widths = (widths / n_grid).view(-1, 1, 1)
                    grid_heights = (heights / n_grid).view(-1, 1, 1)

                    grid_x = box[:, 0].view(-1, 1, 1) + (xv + 0.5).unsqueeze(0) * grid_widths
                    grid_y = box[:, 1].view(-1, 1, 1) + (yv + 0.5).unsqueeze(0) * grid_heights

                    grid_coordinates = torch.stack((grid_x, grid_y), dim=3)  # (n, n_grid, n_grid, 2)

                    grid_coordinates[:, :, :, 0] = grid_coordinates[:, :, :, 0] / img_w * 2 - 1
                    grid_coordinates[:, :, :, 1] = grid_coordinates[:, :, :, 1] / img_h * 2 - 1

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
            gts.append(temp_gts)
            _col_gts_coords.append(temp__col_gts_coords)
        del temp_sfeatures_list, temp_cxcys, temp_img_hws, temp_gts, temp__col_gts_coords
        
        max_nbox_per_frame, _ = n_boxes.max(axis=0)
        start_coord = torch.cumsum(max_nbox_per_frame, dim=0)
        start_coord = torch.cat([torch.tensor([0]).to(device), start_coord]) + 1
        start_coord = start_coord.tolist()
        
        inputs = torch.zeros(batch_size, 1 + max_nbox_per_frame.sum(), len_sampled_feature, device=device)
        inputs_xy_pos = torch.zeros(batch_size, 1 + max_nbox_per_frame.sum(), dim, device=device)
        inputs_frame_pos = torch.zeros_like(inputs_xy_pos)
        inputs_mask = torch.zeros(batch_size, 1 + max_nbox_per_frame.sum(), dtype=torch.float32, device=device)
        
        col_gts_coords = []
        for bidx in range(batch_size):
            temp_col_gts_coords = []
            for coords in _col_gts_coords[bidx]:
                temp_list = []
                for coord in coords:
                    if coord[0] != -1:
                        temp_list.append(start_coord[coord[0]] + coord[1].item())
                temp_col_gts_coords.append(temp_list)
            col_gts_coords.append(temp_col_gts_coords)
            
        for bidx in range(batch_size):
            id_coords = id_coords_list[bidx]
            for id in id_coords.keys():
                id_coords[id] = [start_coord[coord[0]] + coord[1] for coord in id_coords[id]]
            
        frame_pos = frame_pe(torch.arange(n_frames, device=device, dtype=torch.float32))
        for fidx in range(n_frames):
            for bidx in range(batch_size):
                inputs[bidx, start_coord[fidx]:start_coord[fidx] + n_boxes[bidx, fidx]] = sfeatures_list[bidx][fidx]
                inputs_frame_pos[bidx, start_coord[fidx]:start_coord[fidx] + n_boxes[bidx, fidx]] = frame_pos[fidx]
                inputs_mask[bidx, start_coord[fidx]:start_coord[fidx] + n_boxes[bidx, fidx]] = 1
                cxcy = cxcys[bidx][fidx]
                img_h, img_w = img_hws[bidx][fidx]
                if cxcy.shape[0] > 0:
                    inputs_xy_pos[bidx, start_coord[fidx]:start_coord[fidx] + n_boxes[bidx, fidx]] = xy_pe(cxcy[:, 0], cxcy[:, 1], img_h, img_w)


        # attention mask
        atten_mask = (inputs_mask.unsqueeze(dim=2) @ inputs_mask.unsqueeze(dim=1)).bool()

        for fidx in range(n_frames):
            atten_mask[:, start_coord[fidx]:start_coord[fidx + 1], start_coord[fidx]:start_coord[fidx + 1]] = 0
        atten_mask.tril_(diagonal=0)

        atten_mask[:, torch.arange(atten_mask.shape[1]), torch.arange(atten_mask.shape[1])] = 1
        atten_mask[:, :, 0] = 1

        atten_mask = atten_mask.unsqueeze(dim=1)

        h = model(inputs, inputs_frame_pos, inputs_xy_pos, atten_mask)
        del inputs, inputs_frame_pos, inputs_xy_pos, atten_mask
        loss = 0.
        for bidx in range(batch_size):
            for _fidx in range(n_frames - 1):
                gt = gts[bidx][_fidx].to(device)
                if gt.shape[0] == 0:
                    continue
                temp = h[bidx, col_gts_coords[bidx][_fidx]] @ h[bidx, start_coord[_fidx + 1]:start_coord[_fidx + 1] + n_boxes[bidx, _fidx + 1]].T
                celoss = CEloss(temp, gt)
                loss += celoss
        del gt, temp
        loss /= (batch_size * (n_frames - 1))
        
        # Metrics log
        if isinstance(loss, torch.Tensor):
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

        del h, loss
        
        torch.cuda.empty_cache()

        # For logging
        iter_end_timestamp = time.time()
        metric_log.update(name="time per iter", value=iter_end_timestamp-iter_start_timestamp)
        iter_start_timestamp = iter_end_timestamp
        # Outputs logs
        if _i % config["ACCUMULATION_STEPS"] == 0:
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




