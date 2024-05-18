import os
import argparse
import torch.distributed
import torch.backends.cuda
import torch.backends.cudnn

from utils.utils import distributed_rank
from utils.utils import yaml_to_dict
from configs.utils import update_config


def parse_option():
    parser = argparse.ArgumentParser("Network training and evaluation script.", add_help=True)

    parser.add_argument("--git-version", type=str)

    # About system, Like GPUs:
    parser.add_argument("--available-gpus", type=str, help="Available GPUs, like '0,1,2,3'.")
    parser.add_argument("--use-distributed", action="store_true", help="Use distributed training.")

    # Only For **Result Submit Process**:
    parser.add_argument("--submit-model", type=str)
    parser.add_argument("--submit-data-split", type=str)

    # Pretrained Model Load:
    parser.add_argument("--pretrained-model", type=str, help="Pretrained model path.")
    # Resume
    parser.add_argument("--resume", type=str, help="Resume checkpoint path.")
    parser.add_argument("--resume-scheduler", type=str, help="Whether resume the training scheduler.")

    # About Paths:
    # Config file:
    parser.add_argument("--config-path", type=str, help="Config file path.")
    # Data Path:
    parser.add_argument("--data-root", type=str, help="Dataset root dir.", default="./datasets")
    # Log outputs:
    parser.add_argument("--outputs-dir", type=str, help="Outputs dir path.")

    return parser.parse_args()


def main(config: dict):
    os.environ["CUDA_VISIBLE_DEVICES"] = config["AVAILABLE_GPUS"]

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if config["USE_DISTRIBUTED"]:
        torch.distributed.init_process_group("nccl")
        torch.cuda.set_device(distributed_rank())

    from train_engine import train
    from submit_engine import submit

    if config["MODE"] == "train":
        train(config=config)
    elif config["MODE"] == "submit":
        submit(config=config)
    else:
        raise ValueError(f"Unsupported mode '{config['MODE']}'")
    return


if __name__ == '__main__':
    opt = parse_option()                  # runtime options
    cfg = yaml_to_dict(opt.config_path)   # configs

    # Merge parser option and .yaml config, then run main function.
    merged_config = update_config(config=cfg, option=opt)
    merged_config["CONFIG_PATH"] = opt.config_path
    main(config=merged_config)
