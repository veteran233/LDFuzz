import numpy as np
import torch
from torch.utils.data import Dataset
from ...common.laserscan import LaserScan


class SemanticKitti(Dataset):

    def __init__(
        self,  # directory where data is
        from_point_cloud,  # point cloud
        color_map,  # colors dict bgr (e.g 10: [255, 0, 0])
        learning_map,  # classes to learn (0 to N-1 for xentropy)
        learning_map_inv,  # inverse of previous (recover labels)
        sensor,  # sensor to parse scans from
        max_points=150000):  # max number of points present in dataset
        # save deats
        self.from_point_cloud = from_point_cloud
        self.color_map = color_map
        self.learning_map = learning_map
        self.learning_map_inv = learning_map_inv
        self.sensor = sensor
        self.sensor_img_H = sensor["img_prop"]["height"]
        self.sensor_img_W = sensor["img_prop"]["width"]
        self.sensor_img_means = torch.tensor(sensor["img_means"],
                                             dtype=torch.float)
        self.sensor_img_stds = torch.tensor(sensor["img_stds"],
                                            dtype=torch.float)
        self.sensor_fov_up = sensor["fov_up"]
        self.sensor_fov_down = sensor["fov_down"]
        self.max_points = max_points

        # get number of classes (can't be len(self.learning_map) because there
        # are multiple repeated entries, so the number that matters is how many
        # there are for the xentropy)
        self.nclasses = len(self.learning_map_inv)

        # sanity checks
        # make sure color_map is a dict
        assert (isinstance(self.color_map, dict))

        # make sure learning_map is a dict
        assert (isinstance(self.learning_map, dict))

    def __getitem__(self, index):
        # open a semantic laserscan
        scan = LaserScan(project=True,
                         H=self.sensor_img_H,
                         W=self.sensor_img_W,
                         fov_up=self.sensor_fov_up,
                         fov_down=self.sensor_fov_down)

        # open and obtain scan
        scan.open_scan(self.from_point_cloud)

        # make a tensor of the uncompressed data (with the max num points)
        unproj_n_points = scan.points.shape[0]
        unproj_xyz = torch.full((self.max_points, 3), -1.0, dtype=torch.float)
        unproj_xyz[:unproj_n_points] = torch.from_numpy(scan.points)
        unproj_range = torch.full([self.max_points], -1.0, dtype=torch.float)
        unproj_range[:unproj_n_points] = torch.from_numpy(scan.unproj_range)
        unproj_remissions = torch.full([self.max_points],
                                       -1.0,
                                       dtype=torch.float)
        unproj_remissions[:unproj_n_points] = torch.from_numpy(scan.remissions)
        unproj_labels = []

        # get points and labels
        proj_range = torch.from_numpy(scan.proj_range).clone()
        proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
        proj_remission = torch.from_numpy(scan.proj_remission).clone()
        proj_mask = torch.from_numpy(scan.proj_mask)
        proj_labels = []
        proj_x = torch.full([self.max_points], -1, dtype=torch.long)
        proj_x[:unproj_n_points] = torch.from_numpy(scan.proj_x)
        proj_y = torch.full([self.max_points], -1, dtype=torch.long)
        proj_y[:unproj_n_points] = torch.from_numpy(scan.proj_y)
        proj = torch.cat([
            proj_range.unsqueeze(0).clone(),
            proj_xyz.clone().permute(2, 0, 1),
            proj_remission.unsqueeze(0).clone()
        ])
        proj = (proj - self.sensor_img_means[:, None, None]
                ) / self.sensor_img_stds[:, None, None]
        proj = proj * proj_mask.float()

        # return
        return proj, proj_mask, proj_labels, unproj_labels, proj_x, proj_y, proj_range, unproj_range, proj_xyz, unproj_xyz, proj_remission, unproj_remissions, unproj_n_points

    def __len__(self):
        return 1

    @staticmethod
    def map(label, mapdict):
        # put label from original values to xentropy
        # or vice-versa, depending on dictionary values
        # make learning map a lookup table
        maxkey = 0
        for key, data in mapdict.items():
            if isinstance(data, list):
                nel = len(data)
            else:
                nel = 1
            if key > maxkey:
                maxkey = key
        # +100 hack making lut bigger just in case there are unknown labels
        if nel > 1:
            lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
        else:
            lut = np.zeros((maxkey + 100), dtype=np.int32)
        for key, data in mapdict.items():
            try:
                lut[key] = data
            except IndexError:
                print("Wrong key ", key)
        # do the mapping
        return lut[label]


class Parser():
    # standard conv, BN, relu
    def __init__(
        self,
        color_map,  # color for each label
        learning_map,  # mapping for training labels
        learning_map_inv,  # recover labels from xentropy
        sensor,  # sensor to use
        max_points,  # max points in each scan in entire dataset
        batch_size,  # batch size for train and val
        workers,
    ):  # threads to load data
        super(Parser, self).__init__()

        # if I am training, get the dataset
        self.color_map = color_map
        self.learning_map = learning_map
        self.learning_map_inv = learning_map_inv
        self.sensor = sensor
        self.max_points = max_points
        self.batch_size = batch_size
        self.workers = workers

        # number of classes that matters is the one for xentropy
        self.nclasses = len(self.learning_map_inv)

        # Data loading code
        def testloader(from_point_cloud):
            test_dataset = SemanticKitti(
                from_point_cloud=from_point_cloud,
                color_map=self.color_map,
                learning_map=self.learning_map,
                learning_map_inv=self.learning_map_inv,
                sensor=self.sensor,
                max_points=max_points)
            return torch.utils.data.DataLoader(test_dataset,
                                               batch_size=self.batch_size,
                                               shuffle=False,
                                               num_workers=self.workers,
                                               pin_memory=True,
                                               drop_last=True)

        self.testloader = testloader

    def get_test_set(self, from_point_cloud):
        return self.testloader(from_point_cloud)

    def get_n_classes(self):
        return self.nclasses

    def to_original(self, label):
        # put label in original values
        return SemanticKitti.map(label, self.learning_map_inv)

    def to_xentropy(self, label):
        # put label in xentropy values
        return SemanticKitti.map(label, self.learning_map)

    def to_color(self, label):
        # put label in original values
        label = SemanticKitti.map(label, self.learning_map_inv)
        # put label in color
        return SemanticKitti.map(label, self.color_map)
