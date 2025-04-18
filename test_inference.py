import torch
import logging
from pathlib import Path

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import build_dataloader

from pcdet.datasets.custom.custom_dataset import CustomDataset

from inference_utils import run_inference


cfg_file = 'tools/cfgs/kitti_models/pointpillar.yaml'
ckpt_file = '/home/priscilla/Downloads/pointpillar_7728.pth'
custom_data_root = Path('data/kitti/ImageSets') 
cfg_from_yaml_file(cfg_file, cfg)

logger = logging.getLogger("OpenPCDet")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

dataset = CustomDataset(
    dataset_cfg=cfg.DATA_CONFIG,
    class_names=cfg.CLASS_NAMES,
    training=False,
    root_path=custom_data_root,
    logger=logger
)
class_names = dataset.class_names

model = build_network(
    model_cfg=cfg.MODEL,
    num_class=len(class_names),
    dataset=dataset
)
model.load_params_from_file(ckpt_file, logger=logger)
model.cuda()
model.eval()

dataloader, _, _ = build_dataloader(
    dataset_cfg=cfg.DATA_CONFIG,
    class_names=class_names,
    batch_size=1,
    dist=False,
    training=False,
    logger=logger
)

data_batch = next(iter(dataloader))
load_data_to_gpu(data_batch)

results = run_inference(
    model=model,
    data_batch=data_batch,
    class_names=class_names,
    return_probs=True,
    save_path="results_with_probs.json"
)
