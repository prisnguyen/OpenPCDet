import numpy as np
import torch
from pathlib import Path
import argparse
import os
import glob

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import DatasetTemplate
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names,
            training=training, root_path=root_path, logger=logger
        )

        self.root_path = root_path
        self.ext = ext

        if self.root_path.is_dir():
            data_file_list = glob.glob(str(root_path / f'*{self.ext}'))
        else:
            data_file_list = [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError(f"Unsupported extension: {self.ext}")

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)

        if 'voxel_coords' in data_dict and data_dict['voxel_coords'].shape[1] == 3:
            coords = data_dict['voxel_coords']
            batch_idx = np.zeros((coords.shape[0], 1), dtype=np.int32)
            data_dict['voxel_coords'] = np.hstack((batch_idx, coords))

        data_dict['batch_size'] = 1
        return data_dict


def load_model(cfg_path, ckpt_path, device):
    cfg_from_yaml_file(cfg_path, cfg)

    class_names = cfg.CLASS_NAMES if hasattr(cfg, 'CLASS_NAMES') else ['Car', 'Pedestrian', 'Cyclist']
    dummy_dataset = type('Dummy', (), {
        'class_names': class_names,
        'dataset_cfg': cfg.DATA_CONFIG,
        'point_feature_encoder': type('DummyEncoder', (), {
            'num_point_features': getattr(cfg.DATA_CONFIG, 'NUM_POINT_FEATURES', 4)
        })(),
        'grid_size': np.array(getattr(cfg.DATA_CONFIG, 'GRID_SIZE', [432, 496, 1])),
        'voxel_size': getattr(cfg.DATA_CONFIG, 'VOXEL_SIZE', [0.05, 0.05, 0.1]),
        'point_cloud_range': getattr(cfg.DATA_CONFIG, 'POINT_CLOUD_RANGE', [0, -40, -3, 70.4, 40, 1]),
        'depth_downsample_factor': getattr(cfg.MODEL, 'DEPTH_DOWNSAMPLE_FACTOR', 1),
    })()


    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(class_names),
        dataset=dummy_dataset
    )

    logger = common_utils.create_logger()
    model.load_params_from_file(
        filename=ckpt_path,
        to_cpu=(device == 'cpu'),
        logger=logger
    )
    model.to(device).eval()
    return model, cfg


@torch.no_grad()
def extract_embedding(model, data_dict):
    model.eval()
    load_data_to_gpu(data_dict)
    model(data_dict)

    print("✅ data_dict keys after model forward:", list(data_dict.keys()))
    print("✅ shape of spatial_features:", data_dict.get("spatial_features", None).shape if "spatial_features" in data_dict else "not found")

    if 'spatial_features' in data_dict:
        return data_dict['spatial_features']
    
    elif 'point_features' in data_dict:
        return data_dict['point_features']

    raise KeyError("No known embedding feature found in data_dict.")


def main():
    parser = argparse.ArgumentParser(description="General Embedding Generator")
    parser.add_argument('--cfg_file', required=True, help='Path to model config file')
    parser.add_argument('--ckpt_file', required=True, help='Path to checkpoint file')
    parser.add_argument('--input_file', required=True, help='Path to .bin or .npy file')
    parser.add_argument('--output_file', default='embedding.pt', help='Output path for embedding')
    parser.add_argument('--layer', default='spatial_features', help='Layer name to extract (currently fixed to spatial_features)')
    parser.add_argument('--ext', default='.bin', help='Extension of the input file')

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, cfg_loaded = load_model(args.cfg_file, args.ckpt_file, device)

    dataset = DemoDataset(
        dataset_cfg=cfg_loaded.DATA_CONFIG,
        class_names=cfg_loaded.CLASS_NAMES,
        training=False,
        root_path=Path(args.input_file).parent,
        ext=args.ext
    )

    data_dict = dataset[0]
    embedding = extract_embedding(model, data_dict)
    torch.save(embedding, args.output_file)
    print(f"Embedding saved to: {args.output_file}")
    print(f"Shape: {tuple(embedding.shape)}")


if __name__ == '__main__':
    main()