# _*_coding:utf-8_*_

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys

import argparse, pickle
import os

os.environ['PROJECT_DIR'] = '/home/szw/code/r_deephunter'

# from keras.models import load_model
from utils.KITTI_LoadModel import load_model, load_frd_model
# import tensorflow as tf
# from keras.utils import CustomObjectScope
from tqdm import tqdm

from _lib.fuzzer import Fuzzer
from _lib.queue.seed import Seed
# from statistics import draw
from utils import DUMPS_utlis
from utils.config import metrics_para, get_seed_path, get_output_path
from _lib.func import metadata_function, iterate_function, build_objective_function, build_fetch_function, build_frd_function, velodyne_mutation_function_2

sys.path.append('../')

# from keras import Input

# from keras.applications import MobileNet, VGG19, ResNet50

import random
import time
from datetime import datetime as dt
from _lib.queue.queue_coverage import ImageInputCorpus


# 初始队列里只有10个数据
def dry_run(indir, fetch_function, queue, batch_num):
    seed_lis = os.listdir(indir)
    # Read each initial seed and analyze the coverage
    # for seed_name in tqdm(seed_lis):
    # progress = iter(seed_lis)
    DUMPS_utlis.init_DUMPS(queue.DUMPS)
    with tqdm(total=len(seed_lis)) as progress_bar:
        for __i__ in range(0, len(seed_lis), batch_num):
            # tf.logging.info("Attempting dry run with '%s'...", seed_name)
            img = []
            for __cnt__ in range(__i__, min(len(seed_lis), __i__ + batch_num)):
                seed_name = seed_lis[__cnt__]
                path = os.path.join(indir, seed_name)
                with open(path, 'rb') as f:
                    _temp = pickle.load(f)
                    if not isinstance(_temp, list): _temp = [_temp]
                    img += _temp
                    progress_bar.update(1)

            # Each seed will contain two images, i.e., the reference image and mutant (see the paper)
            input_batches = img
            # Predict the mutant and obtain the outputs
            # coverage_batches is the output of internal layers and metadata_batches is the output of the prediction result
            _, metadata_batches = fetch_function(input_batches)
            # Based on the output, compute the coverage information
            metadata_list = metadata_function(metadata_batches)
            # coverage_list = coverage_function(coverage_batches, lb=metadata_list)
            # coverage_list = np.random.rand(1000)
            # coverage_list = coverage_list * 2
            # coverage_list = coverage_list.astype(dtype=np.uint8)
            # Create a new seed

            for __cnt__ in range(0, batch_num):
                if __i__ + __cnt__ >= len(seed_lis):
                    break

                seed_name = seed_lis[__i__ + __cnt__]
                scene = seed_name.split('.')[0]

                input = Seed(seed_name, None)

                DUMPS_utlis.updateBatches_DUMPS(queue.DUMPS, seed_name,
                                                input_batches[__cnt__],
                                                metadata_list[__cnt__])

                queue.DUMPS['frd_limit'] += input_batches[__cnt__]['frd_limit']

                queue.save_if_interesting(input, input_batches[__cnt__], False,
                                          True, scene)

    DUMPS_utlis.updateCoverage_DUMPS(queue.DUMPS)

    queue.DUMPS['frd_limit'] /= len(queue.DUMPS['scene'])


# 获得实验模型
# def get_model(args):
#     img_rows, img_cols = 224, 224
#     input_shape = (img_rows, img_cols, 3)
#     input_tensor = Input(shape=input_shape)
#     model = None
#     if args.model == 'mobilenet':
#         model = MobileNet(input_tensor=input_tensor)
#     elif args.model == 'vgg19':
#         model = VGG19(input_tensor=input_tensor)
#     elif args.model == 'resnet50':
#         model = ResNet50(input_tensor=input_tensor)
#     else:
#         model_path = get_model_weight_path(args.data_name, args.model)
#         print(model_path)
#         model = load_model(model_path)
#     return model


# 覆盖指标
def get_cri(args):
    if args.metric_para is None:
        cri = metrics_para[args.criteria]
    elif args.criteria == 'nc':
        cri = args.metric_para
    else:
        cri = int(args.metric_para)
    return cri


def get_queue(args):
    # The seed queue
    # if args.criteria == 'fann':
    #     queue = TensorInputCorpus(args.o, args.random, args.select, cri, "kdtree")
    # else:
    queue = ImageInputCorpus(args.o, args.random, args.select, args.criteria,
                             args.check_point, args.DUMPS)
    return queue


# function
def get_func(args):
    # if args.quantize_test == 1:
    #     model_names = os.listdir(args.quan_model_dir)
    #     model_paths = [os.path.join(args.quan_model_dir, name) for name in model_names]
    #     if args.model == 'mobilenet':
    #         import keras
    #         with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,
    #                                 'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
    #             models = [load_model(m) for m in model_paths]
    #     else:
    #         models = [load_model(m) for m in model_paths]
    #     # fetch_function = build_fetch_function(coverage_handler, preprocess, models)
    #     model_names.insert(0, args.model)
    # else:
    #     # fetch_function = build_fetch_function(coverage_handler, preprocess)
    #     model_names = [args.model]

    # Like AFL, dry_run will run all initial seeds and keep all initial seeds in the seed queue

    # The function to update coverage
    # coverage_function = coverage_handler.update_coverage
    # The function to perform the mutation from one seed
    # mutation_function = image_mutation_function(args.batch_num)  # 不用改
    fetch_function = build_fetch_function(args.model, args.loader)
    mutation_function = velodyne_mutation_function()
    return fetch_function, mutation_function


def execute(args):
    # Get the layers which will be excluded during the coverage computation
    # exclude_layer_list = execlude_layer_dic[args.model]

    # Create the output directory including seed queue and crash dir, it is like AFL
    # if os.path.exists(args.o):
    #     shutil.rmtree(args.o)
    os.makedirs(os.path.join(args.o, 'queue'))
    os.makedirs(os.path.join(args.o, 'crashes'))
    os.makedirs(os.path.join(args.o, 'result'))

    # Load model. For ImageNet, we use the default models from Keras framework.
    # For other models, we load the model from the h5 file.
    # Get the preprocess function based on different dataset
    # preprocess = get_preprocess(args.data_name)

    # Load the profiling information which is needed by the metrics in DeepGauge
    # model_profile_path = get_model_profile_path(data_name, model_name)
    # profile_dict = pickle.load(open(model_profile_path, 'rb'))  # 均值,方差, 不知道,下边界,上边界
    # print(profile_dict)
    # Load the configuration for the selected metrics.
    # cri = get_cri(args)

    # The coverage computer
    # if args.criteria == 'space':
    #     coverage_handler = SpaceCoverage(model=model, k=cri)
    # elif args.criteria == "lsc":
    #     coverage_handler = LSCoverage(model=model, data_name=args.data_name, model_name=args.model_name, k=cri)
    # elif args.criteria == "fake":
    #     coverage_handler = FakeCoverage(model=model)
    # else:
    # coverage_handler = Coverage(model=model, criteria=args.criteria, k=cri,
    #                             profiling_dict=profile_dict, exclude_layer=exclude_layer_list)

    # The log file which records the plot data after each iteration of the fuzzing
    # plot_file = open(os.path.join(args.o, 'plot.log'), 'a+')

    # If testing for quantization, we will load the quantized versions
    # fetch_function is to perform the prediction and obtain the outputs of each layers

    queue = get_queue(args)

    fetch_function = build_fetch_function(args.model, args.loader)
    mutation_function = velodyne_mutation_function_2(queue)

    # Perform the dry_run process from the initial seeds
    dry_run_fetch = fetch_function
    dry_run(args.i, dry_run_fetch, queue, args.batch_num)  # dry_run 初始化队列

    # For each seed, compute the coverage and check whether it is a "bug", i.e., adversarial example
    objective_function = build_objective_function(args)
    frd_function = build_frd_function(args.frd_model)

    # The main fuzzer class
    fuzzer = Fuzzer(queue, metadata_function,
                    objective_function, mutation_function, fetch_function,
                    iterate_function(args), frd_function, args.select)

    # The fuzzing process
    fuzzer.loop(args.max_iteration)
    return time.time()


def param2txt(file_path, msg, mode="a+"):
    f = open(file_path, mode)
    f.write(msg)
    f.close


if __name__ == '__main__':
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    start_time = time.time()

    # tf.logging.set_verbosity(tf.logging.INFO)
    random.seed(time.time())

    parser = argparse.ArgumentParser()
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
                        help='max iteration, default : 1000',
                        type=int,
                        default=1000)

    now = str(dt.strftime(dt.now(), '%Y-%m-%d_%H-%M-%S_%f'))
    data_name = 'kitti'
    args = parser.parse_args()

    model_list = args.m
    guidance_criteria_list = args.gc
    seed_selection_strategy_list = args.s

    for model_name in model_list:
        batch_num = 1
        is_random = 1
        input_dir = get_seed_path(data_name, model_name)

        args.data_name = data_name
        args.model_name = model_name

        args.model, args.loader = load_model(args.model_name)
        args.frd_model = load_frd_model()

        args.now = now
        args.random = is_random
        args.i = input_dir
        args.max_iteration = args.mi
        args.check_point = args.mi // 10
        args.batch_num = batch_num

        for criteria in guidance_criteria_list:
            for select in seed_selection_strategy_list:
                args.criteria = criteria
                args.select = select

                output_dir = get_output_path(args.data_name,
                                             args.model_name,
                                             args.select,
                                             args.criteria,
                                             num=0)

                output_dir = args.now + output_dir

                os.makedirs(output_dir)
                f = open(f'{output_dir}/out.log', 'w')
                sys_out = sys.stdout
                sys.stdout = f

                args.o = output_dir
                args.DUMPS = {'criteria': args.criteria}
                args.iou_threshold = 0.7
                print(args)
                start_time = time.time()
                end_time = execute(args)
                ftime = end_time - start_time
                print('finish', ftime)

                sys.stdout = sys_out
                f.close()
