# _*_coding:utf-8_*_

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys

import argparse, pickle
import os

os.environ['PROJECT_DIR'] = os.path.dirname(__file__)

from tqdm import tqdm

from _lib.fuzzer import Fuzzer
from _lib.queue.seed import Seed

from utils import DUMPS_utlis
from utils.config import get_seed_path, get_output_path
from _lib.func import iterate_function, build_objective_function, build_fetch_function, build_frd_function, lidar_mutation_functionV2

import random
import time
from datetime import datetime as dt
from _lib.queue.queue_coverage import ImageInputCorpus
from config.lidar_config import load_lidar_config


def dry_run(dataset_name, indir, fetch_function, queue, batch_size):
    seed_list = sorted(os.listdir(indir))
    DUMPS_utlis.init_DUMPS(queue.DUMPS)

    with tqdm(total=len(seed_list)) as progress_bar:
        for start_index in range(0, len(seed_list), batch_size):
            infos_dict = []
            for index in range(start_index,
                               min(len(seed_list), start_index + batch_size)):
                seed_name = seed_list[index]
                path = os.path.join(indir, seed_name)
                with open(path, 'rb') as f:
                    _seed = pickle.load(f)
                    if not isinstance(_seed, list): _seed = [_seed]
                    infos_dict += _seed
                    progress_bar.update(1)

            # Each seed will contain two images, i.e., the reference image and mutant (see the paper)
            # Predict the mutant and obtain the outputs
            # coverage_batches is the output of internal layers and metadata_batches is the output of the prediction result
            pred_boxes, _, _ = fetch_function(infos_dict)

            for index in range(0, batch_size):
                if start_index + index >= len(seed_list):
                    break

                seed_name = seed_list[start_index + index]
                scene = seed_name.split('.')[0]

                input = Seed(seed_name, None)

                DUMPS_utlis.updateBatches_DUMPS(queue.DUMPS, dataset_name,
                                                seed_name, infos_dict[index],
                                                pred_boxes[index])

                queue.DUMPS['frd_limit'] += infos_dict[index]['frd_limit']
                infos_dict[index]['frd_score'] = 0

                queue.save_if_interesting(input, infos_dict[index], False,
                                          True, scene)

    DUMPS_utlis.updateCoverage_DUMPS(queue.DUMPS)

    queue.DUMPS['frd_limit'] /= len(queue.DUMPS['scene'])


def get_queue(args):
    queue = ImageInputCorpus(args.output_dir, args.select, args.criteria,
                             args.check_point, args.DUMPS)
    return queue


def execute(args):

    # Create the output directory including seed queue and crash dir, it is like AFL
    os.makedirs(os.path.join(args.output_dir, 'queue'))
    os.makedirs(os.path.join(args.output_dir, 'crashes'))
    os.makedirs(os.path.join(args.output_dir, 'result'))

    queue = get_queue(args)

    load_lidar_config(args.dataset_name)

    fetch_function = build_fetch_function(args.dataset_name, args.model_name,
                                          args.batch_size)

    frd_function = build_frd_function(args.dataset_name)

    mutation_function = lidar_mutation_functionV2(queue, args.dataset_name,
                                                  args.criteria)

    # Perform the dry_run process from the initial seeds
    dry_run(args.dataset_name, args.seed_path, fetch_function, queue,
            args.batch_size)

    # For each seed, compute the coverage and check whether it is a "bug", i.e., adversarial example
    objective_function = build_objective_function(args)

    # The main fuzzer class
    fuzzer = Fuzzer(queue,
                    objective_function, mutation_function, fetch_function,
                    iterate_function(args), frd_function, args.select)

    # The fuzzing process
    fuzzer.loop(args.max_iteration)
    return time.time()


if __name__ == '__main__':
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    start_time = time.time()

    random.seed(time.time())

    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        nargs='+',
                        help='Dataset list, e.g. kitti nuscenes',
                        required=True)
    parser.add_argument(
        '-m',
        nargs='+',
        help='model list, split by blankspace, e.g. pointpillar pv_rcnn',
        required=True)
    parser.add_argument(
        '-gc',
        nargs='+',
        help='guidance criteria list, split by comma, e.g. spc sec',
        required=True)
    parser.add_argument(
        '-s',
        nargs='+',
        help='seed selection strategy list, split by comma, e.g. new,random',
        required=True)

    parser.add_argument('-mi',
                        '--max-iteration',
                        help='max iteration, default : 1000',
                        type=int,
                        default=1000)
    parser.add_argument('--batch-size', help='', type=int, default=2)
    parser.add_argument('--iou-threshold', help='', type=float, default=0.7)

    parser.add_argument('--use-distance', help='', action='store_true')
    parser.add_argument('--distance-threshold',
                        help='',
                        type=float,
                        default=1.0)

    args = parser.parse_args()

    now = str(dt.strftime(dt.now(), '%Y-%m-%d_%H-%M-%S_%f'))

    args.now = now
    args.check_point = args.max_iteration // 10

    dataset_list = args.d
    model_list = args.m
    guidance_criteria_list = args.gc
    seed_selection_strategy_list = args.s

    for dataset_name in dataset_list:
        for model_name in model_list:

            args.dataset_name = dataset_name
            args.model_name = model_name
            args.seed_path = get_seed_path(dataset_name, model_name)

            for select in seed_selection_strategy_list:
                for criteria in guidance_criteria_list:

                    args.select = select
                    args.criteria = criteria

                    output_dir = get_output_path(dataset_name, model_name,
                                                 select, criteria)

                    output_dir = now + output_dir

                    os.makedirs(output_dir)
                    f = open(f'{output_dir}/out.log', 'w')
                    sys_out = sys.stdout
                    sys.stdout = f

                    print(args)

                    args.output_dir = output_dir
                    args.DUMPS = {'criteria': criteria}

                    start_time = time.time()
                    end_time = execute(args)
                    ftime = end_time - start_time

                    print('finish', ftime)

                    sys.stdout = sys_out
                    f.close()
