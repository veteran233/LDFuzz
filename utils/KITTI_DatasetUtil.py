# from pcdet.datasets.kitti.kitti_dataset import KittiDataset
# from easydict import EasyDict
# from pathlib import Path
# import yaml


# root_path = Path("./data/kitti")

# dataset_cfg = EasyDict(yaml.safe_load(
#     stream=open(root_path / "kitti_dataset.yaml")
# ))

# dataset_train = KittiDataset(
#     dataset_cfg=dataset_cfg,
#     class_names=['Car', 'Pedestrian', 'Cyclist'],
#     training=True,
#     root_path=root_path
# )

# dataset_train = KittiDataset(
#     dataset_cfg=dataset_cfg,
#     class_names=['Car', 'Pedestrian', 'Cyclist'],
#     training=False,
#     root_path=root_path
# )


# def merge_lidar():
#     res = []
#     # for d in os.listdir(root_path/"training/velodyne"):
#     #     res.append(dataset_train.get_lidar(d[:-4]).tolist())
#     for i in range(0, 10):
#         res.append(dataset_train.get_lidar("{:06d}".format(i)))
#     return res


# def merge_label():
#     res = []
#     # for d in os.listdir(root_path/"training/label_2"):
#     #     res.append(dataset_train.get_label(d[:-4]).tolist())
#     for i in range(20, 30):
#         res.append(dataset_train.get_label("{:06d}".format(i)))
#     return res


# def load_data():
#     return (
#         None,
#         None
#     ), (
#         merge_lidar(),
#         merge_label()
#     )
