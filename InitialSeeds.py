#!/usr/bin/env python2.7
import os

os.environ['PROJECT_DIR'] = '/home/szw/code/r_deephunter'

from utils.KITTI_LoadModel import load_model, modify_kitti_infos
import numpy as np
import pickle
import copy

from _lib.func import velodyne_mutation_function, build_frd_function

from utils import config, KITTI_LoadModel
from utils.config import get_seed_path

from _others.fid.lidargen_fid import get_fid

frd_function = build_frd_function(KITTI_LoadModel.load_frd_model())
mutation_function = velodyne_mutation_function(no_seed=True)


def max_frd(batch) -> float:
    l = []
    for _i in range(3):
        test = copy.deepcopy(batch)

        for _j in range(10):
            test = mutation_function(test, config.object_level)[0]

        x = frd_function(batch['points'])
        y = frd_function(test['points'])

        frd = get_fid(x, y)
        l.append(frd)

    return np.max(l)


def createBatch(x_batch, batch_size, output_path, prefix):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    batch_num = len(x_batch)
    batches = np.split(x_batch, batch_num, axis=0)
    for i, batch in enumerate(batches):
        # test = np.append(batch, batch, axis=0)
        batch[0]['frd_limit'] = max_frd(batch[0])
        saved_name = f'{prefix}{i:03d}.pickle'
        # np.save(os.path.join(output_path, saved_name), batch)
        with open(os.path.join(output_path, saved_name), 'wb') as f:
            pickle.dump(batch.tolist(), f)


if __name__ == '__main__':

    for data_name, model_name_arr in config.model_data.items():
        for model_name in model_name_arr:

            print(data_name, model_name)
            batch_num = 100
            num_in_each_class = 50
            batch_size = 1

            output_path = get_seed_path(data_name, model_name)

            if not os.path.exists(output_path):
                os.makedirs(output_path)

            model, test_loader = load_model(model_name, batch_size)

            test_loader.dataset.sample_id_list = np.random.choice(
                test_loader.dataset.sample_id_list, batch_num, False).tolist()
            test_loader.dataset.kitti_infos = test_loader.dataset.get_infos()

            # modify kitti_infos
            modify_kitti_infos(test_loader)
            # ----------------------------------------

            test_loader.dataset.kitti_infos = np.array(
                test_loader.dataset.kitti_infos)

            selected = np.random.choice(len(test_loader.dataset.kitti_infos),
                                        num_in_each_class,
                                        replace=False)
            # selected = [i for i in selected]
            createBatch(test_loader.dataset.kitti_infos[selected], batch_size,
                        output_path, 'init')
            print('finish')
