from utils.KITTIDataset import build_dataloader
# from utils.KITTIDataset import KittiDataset
from pcdet.config import merge_new_config
from pcdet.models import build_network

from easydict import EasyDict
from pathlib import Path

from . import config
from .coverage_utils import get_scene_graph_type
from _others.lidar_bonnetal.modules.user import User

import yaml

from mtest.utils import calibration_kitti
from mtest.core.pose_estimulation.road_split import road_split

from pcdet.utils import box_utils, common_utils

from shapely import concave_hull
from shapely.geometry import MultiPoint

from torch.utils.data import DataLoader

import numpy as np

root_path = Path("./data/kitti")

profile_path = Path("./__profile/kitti")

pointpillar_model_cfg = profile_path / "cfg/pointpillar.yaml"

pv_rcnn_model_cfg = profile_path / "cfg/pv_rcnn.yaml"

second_model_cfg = profile_path / "cfg/second.yaml"

pointrcnn_model_cfg = profile_path / "cfg/pointrcnn.yaml"

pointpillar_model_path = profile_path / "models" / (config.pointpillar +
                                                    ".pth")

pv_rccn_model_path = profile_path / "models" / (config.pv_rcnn + ".pth")

second_model_path = profile_path / "models" / (config.second + ".pth")

pointrcnn_model_path = profile_path / "models" / (config.pointrcnn + ".pth")

model_cfg = {
    config.pointpillar: pointpillar_model_cfg,
    config.pv_rcnn: pv_rcnn_model_cfg,
    config.second: second_model_cfg,
    config.pointrcnn: pointrcnn_model_cfg
}

model_path = {
    config.pointpillar: pointpillar_model_path,
    config.pv_rcnn: pv_rccn_model_path,
    config.second: second_model_path,
    config.pointrcnn: pointrcnn_model_path
}


def load_model(model_type: str, batch_size=4, mode=False):

    _model_cfg = merge_new_config(
        config=EasyDict(),
        new_config=EasyDict(
            yaml.safe_load(stream=open(model_cfg[model_type]))))

    _set = _loader = None

    if mode:
        _set, _loader, _ = build_dataloader(dataset_cfg=_model_cfg.DATA_CONFIG,
                                            class_names=_model_cfg.CLASS_NAMES,
                                            batch_size=batch_size,
                                            dist=False,
                                            root_path=root_path,
                                            training=True)
    else:
        _set, _loader, _ = build_dataloader(dataset_cfg=_model_cfg.DATA_CONFIG,
                                            class_names=_model_cfg.CLASS_NAMES,
                                            batch_size=batch_size,
                                            dist=False,
                                            root_path=root_path,
                                            training=False)

    model = build_network(model_cfg=_model_cfg.MODEL,
                          num_class=len(_model_cfg.CLASS_NAMES),
                          dataset=_set)

    model.train(False)
    model.load_params_from_file(
        model_path[model_type],
        common_utils.create_logger(model_type + "_log.log"))

    return model, _loader


def load_frd_model():
    ARCH = yaml.safe_load(
        open('_others/lidar_bonnetal/model/arch_cfg.yaml', 'r'))
    DATA = yaml.safe_load(
        open('_others/lidar_bonnetal/model/data_cfg.yaml', 'r'))
    return User(ARCH, DATA, '_others/lidar_bonnetal/model')


def modify_kitti_infos(loader: DataLoader):
    _i = 0
    while _i < len(loader.dataset.kitti_infos):
        info = loader.dataset.kitti_infos[_i]

        sample_idx = info['point_cloud']['lidar_idx']
        img_shape = info['image']['image_shape']
        annos = info['annos']

        _mask = (annos['name'] != 'DontCare')
        car_mask = (annos['name'] == 'Car')

        for _k in annos:
            if _k == 'gt_boxes_lidar': annos[_k] = annos[_k][car_mask[_mask]]
            else: annos[_k] = annos[_k][car_mask]

        calib = loader.dataset.get_calib(sample_idx)

        points = loader.dataset.get_lidar(sample_idx)

        # FOV
        if loader.dataset.dataset_cfg.FOV_POINTS_ONLY:
            pts_rect = calib.lidar_to_rect(points[:, 0:3])
            fov_mask = loader.dataset.get_fov_flag(pts_rect, img_shape, calib)
            points = points[fov_mask]

        # Point Cloud Range
        range_mask = common_utils.mask_points_by_range(
            points, loader.dataset.point_cloud_range)
        points = points[range_mask]

        # Selected
        selected = common_utils.keep_arrays_by_name(annos['name'], ['Car'])
        corners_lidar = box_utils.boxes_to_corners_3d(
            annos['gt_boxes_lidar'][selected])
        selected_mask = None
        for k in range(corners_lidar.shape[0]):
            selected_mask = \
                np.append(
                    selected_mask,
                    box_utils.in_hull(
                        points[:, 0:3],
                        corners_lidar[k]
                    )[np.newaxis, ...],
                    axis=0
                ) if selected_mask is not None else\
                box_utils.in_hull(
                    points[:, 0:3],
                    corners_lidar[k]
                )[np.newaxis, ...]

        info['selected_name'] = annos['name'][selected]
        info['selected_gt_boxes'] = annos['gt_boxes_lidar'][selected]
        info['selected_pc_mask'] = selected_mask
        info['points'] = points
        info['is_fn'] = np.array([False] * corners_lidar.shape[0])
        info['is_fn2'] = np.array([False] * corners_lidar.shape[0])

        if points.shape[0] == 0:
            loader.dataset.kitti_infos.pop(_i)
            print(f'pop {sample_idx} -- low points')
            continue

        if info['selected_gt_boxes'].shape[0] == 0:
            loader.dataset.kitti_infos.pop(_i)
            print(f'pop {sample_idx} -- low gt boxes')
            continue

        road_pc, road_labels, non_road_pc, idx_road = road_split(
            int(sample_idx), f'{root_path}/training/velodyne/{sample_idx}.bin',
            f'{root_path}/training/road_label', 'log')

        if road_pc.shape[0] < 10:
            loader.dataset.kitti_infos.pop(_i)
            print(f'pop {sample_idx} -- low road_pc')
            continue

        s_rl = road_labels[fov_mask][range_mask].reshape(-1)
        # 40: "road"
        # 44: "parking"
        # 48: "sidewalk"
        s_rl = (s_rl == 44) + (s_rl == 40)
        s_rl_pts = points[s_rl][:, 0:2]

        road_hull = concave_hull(MultiPoint(s_rl_pts), 0.3)
        if road_hull.area < 10:
            loader.dataset.kitti_infos.pop(_i)
            print(f'pop {sample_idx} -- low road_hull')
            continue

        info['weather_type'] = 'sunny'

        info['road_hull'] = road_hull
        info['scene_graph_type'] = get_scene_graph_type(road_hull)
        info['road_pc'] = road_pc
        info['road_labels'] = road_labels
        info['non_road_pc'] = non_road_pc
        info['mtest_calib'] = calibration_kitti.Calibration(
            f'{root_path}/training/calib/{sample_idx}.txt')

        _i += 1
