# _*_coding:utf-8_*_
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import copy
import pickle

import numpy as np
import torch

import shapely
from shapely.geometry import MultiPoint, Polygon, MultiPolygon

from pcdet.ops.iou3d_nms import iou3d_nms_utils

from _lib.queue.queue import FuzzQueue
from _lib.queue.queue_coverage import ImageInputCorpus
from _lib.queue.seed import Seed
from _others.velodyne_mutators import METHOD, WEATHER_METHOD
from utils import config, DUMPS_utlis
from utils.coverage_utils import cal_single_c1, cal_single_c2, get_gt_polygon, get_scene_graph_encode


def metadata_function(meta_batches):
    return meta_batches


def image_mutation_function(batch_num):
    # Given a seed, randomly generate a batch of mutants
    def func(seed):
        return Mutators.image_random_mutate(seed, batch_num)

    return func


def velodyne_mutation_function(no_seed=False):

    def mutate(batch, method_list=None):
        if method_list is None:
            if batch['weather_type'] in config.scene_level:
                method_list = np.array(config.object_level)
            else:
                method_list = np.array(config.object_level +
                                       config.scene_level)
        else:
            if not isinstance(method_list, np.ndarray):
                method_list = np.array(method_list)

        method = np.random.choice(method_list)

        batch['method'] = method
        print('mutation : %s' % method)

        try:
            return METHOD[method](copy.deepcopy(batch))
        except:
            print(
                f'method --{method}-- failed to mutate, which will be choose another.'
            )

            method_list = method_list[method_list != method]
            if len(method_list) == 0:
                method_list = None

            return mutate(batch, method_list)

    def func(seed, method_list=None):
        with open(seed.fname, 'rb') as f:
            batch = pickle.load(f)
        return mutate(batch, method_list)

    def func2(batch, method_list=None):
        return mutate(copy.deepcopy(batch), method_list)

    if no_seed: return func2
    else: return func


def velodyne_mutation_function_2(queue: FuzzQueue):

    def mutate():
        method = np.random.choice(config.object_level + config.scene_level)
        print(f'Select {method}')

        while (True):
            parent = queue.select_next(1)[0]
            with open(parent.fname, 'rb') as f:
                batch = pickle.load(f)

            if method in config.scene_level and batch[
                    'weather_type'] in config.scene_level:
                method = np.random.choice(config.object_level)
                print(f'Select Object-Level : {method}')

            batch['method'] = method

            try:
                return parent, METHOD[method](batch, DUMPS=queue.DUMPS)[0]
            except:
                del batch
                del parent
                continue

    return mutate


def build_objective_function(args):

    def func(seed, data_batch):
        pd_boxes = data_batch['pred_boxes']
        ground_truth = data_batch['selected_gt_boxes']
        method = data_batch['method']

        pred_scores = data_batch['pred_scores']
        gt_scores_list = \
            data_batch['gt_scores_list']\
            if 'gt_scores_list' in data_batch else\
            np.array([2] * ground_truth.shape[0], dtype=np.float32)

        if gt_scores_list.shape[0] < ground_truth.shape[0]:
            gt_scores_list = \
                np.concatenate([
                    gt_scores_list,
                    [2] * (ground_truth.shape[0] - gt_scores_list.shape[0])
                ])

        ref_gt_scores_list = copy.deepcopy(gt_scores_list)

        iou = iou3d_nms_utils.boxes_bev_iou_cpu(ground_truth, pd_boxes)
        iou_gt, iou_pd = np.where(iou >= args.iou_threshold)
        gt_scores_list[iou_gt] = pred_scores[iou_pd]
        all_gt = np.array(list(range(gt_scores_list.shape[0])))
        gt_scores_list[np.delete(all_gt, iou_gt)] = 0
        data_batch['gt_scores_list'] = gt_scores_list

        if iou.shape[1] != 0:
            _iou = copy.deepcopy(iou)
            _iou[_iou < _iou.max(axis=0).reshape(1, -1)] = 0
            _iou[_iou < _iou.max(axis=1).reshape(-1, 1)] = 0

            obj_missing = np.sum(np.max(_iou, axis=1).reshape(-1) <= 0)
            false_det = np.sum(np.max(_iou, axis=0).reshape(-1) <= 0)
            fail_loc = np.sum(
                ((0 < _iou) * (_iou < args.iou_threshold)).reshape(-1))
        else:
            obj_missing = iou.shape[0]
            false_det = 0
            fail_loc = 0

        if iou.shape[1] != 0:
            iou = np.max(iou, axis=1).reshape(-1)
        else:
            iou = np.array([0] * iou.shape[0])

        tp = np.sum(iou >= args.iou_threshold)
        fp = len(pd_boxes) - tp
        fn = len(ground_truth) - tp
        data_batch['iou_tp'] = tp

        print(f'Predict -- TP : {tp} FP : {fp} FN : {fn}')

        fn_list = np.where(iou < args.iou_threshold)[0]
        # fn_list = data_batch['fn_list'] if 'fn_list' in data_batch else None
        # fn_list = \
        #     np.concatenate([
        #         fn_list,
        #         np.where(iou < 0.5)[0]
        #     ]) if fn_list is not None else\
        #     np.where(iou < 0.5)[0]
        # fn_list = np.unique(fn_list)
        # data_batch['fn_list'] = fn_list

        args.DUMPS['fp'][method][-1] += fp
        args.DUMPS['fn'][method][-1] += fn
        args.DUMPS['fn1'][method][-1] += np.sum(
            (data_batch['is_fn'] == False) * (iou < args.iou_threshold))
        data_batch['is_fn'] += iou < args.iou_threshold

        args.DUMPS['fn2'][method][-1] += np.sum(
            (data_batch['is_fn2'] == False) * (iou < args.iou_threshold))
        data_batch['is_fn2'] += iou < args.iou_threshold

        args.DUMPS['obj_missing'][method][-1] += obj_missing
        args.DUMPS['false_det'][method][-1] += false_det
        args.DUMPS['fail_loc'][method][-1] += fail_loc

        s_gt_boxes = data_batch['selected_gt_boxes']
        s_name = data_batch['selected_name']
        weather_type = data_batch['weather_type']
        sg_type = data_batch['scene_graph_type']
        data_batch['level'] += 1

        scene = seed.root_seed.split('.')[0]
        dict_scene = args.DUMPS['scene'][scene]
        dict_cluster = args.DUMPS['cluster'][sg_type]

        dict_scene['method_times'][method] += 1
        dict_scene['total_times'] += 1

        # ## RQ3
        # fn3_count = 0
        # for i_fn in fn_list:
        #     cor = box_utils.boxes_to_corners_3d(s_gt_boxes[np.newaxis, i_fn])
        #     cor = shapely.convex_hull(MultiPoint(cor[0, :4, :2]))
        #     if not dict_scene['error_polygon'].contains(cor):
        #         fn3_count += 1

        #     dict_scene['error_polygon'] = shapely.union(
        #         dict_scene['error_polygon'], cor)

        # args.DUMPS['fn3'][method][-1] += fn3_count
        # dict_cluster['error_cluster'].add(
        #     get_scene_graph_encode(s_gt_boxes[fn_list], s_name[fn_list],
        #                            weather_type, sg_type))
        # ## RQ3 End

        gt_polygon = get_gt_polygon(s_gt_boxes)
        err_polygon = get_gt_polygon(s_gt_boxes[fn_list])
        if err_polygon is None: err_polygon = Polygon()

        sg_encode = get_scene_graph_encode(s_gt_boxes, s_name, weather_type,
                                           sg_type)
        err_sg_encode = get_scene_graph_encode(s_gt_boxes[fn_list],
                                               s_name[fn_list], weather_type,
                                               sg_type)
        # if fn_list.shape[0] > 0: err_sg_encode = sg_encode
        # else: err_sg_encode = 0
        # dict_scene['gt_polygon'] = shapely.union(dict_scene['gt_polygon'],
        #                                          get_gt_polygon(s_gt_boxes))
        # dict_scene['cluster'].add(sg_encode)

        # GLOBAL
        # dict_cluster['cluster'].add(sg_encode)
        # dict_cluster['error_cluster'].add(err_sg_encode)
        # GLOBAL END

        # add to Error Detail
        args.DUMPS['error_detail'].append({
            'scene': scene,
            'scene_graph_type': sg_type,
            'method': method,
            'weather_type': weather_type,
            'gt_boxes': s_gt_boxes,
            'fn_list': fn_list,
            'sg_encode': sg_encode,
            'err_sg_encode': err_sg_encode
        })

        ref_gt_polygon = dict_scene['gt_polygon']
        ref_cluster = dict_cluster['cluster']
        ref_err_polygon = dict_scene['error_polygon']
        ref_err_cluster = dict_cluster['error_cluster']

        # coverage1 = get_scene_coverage1_DUMPSver(args.DUMPS, scene)
        # coverage2 = get_scene_coverage2_DUMPSver(args.DUMPS, scene)
        gt_polygon = shapely.union(ref_gt_polygon, gt_polygon)

        cluster = copy.deepcopy(ref_cluster)
        cluster.add(sg_encode)

        err_polygon = shapely.union(ref_err_polygon, err_polygon)

        err_cluster = copy.deepcopy(ref_err_cluster)
        err_cluster.add(err_sg_encode)

        ref_mean_scores = np.mean(ref_gt_scores_list)
        mean_scores = np.mean(gt_scores_list)

        # if args.criteria == 'c1_confi':  # 第一个覆盖, 置信度
        #     ret = ref_coverage1 < coverage1 or mean_scores <= ref_mean_scores
        # elif args.criteria == 'c2_confi':  # 第二个覆盖, 置信度
        #     ret = ref_coverage2 < coverage2 or mean_scores <= ref_mean_scores
        # elif args.criteria == 'confi':  # 仅置信度
        #     ret = mean_scores <= ref_mean_scores
        # elif args.criteria == 'c1':  # 仅第一个覆盖
        #     ret = ref_coverage1 < coverage1
        # elif args.criteria == 'c2':  # 仅第二个覆盖
        #     ret = ref_coverage2 < coverage2
        # elif args.criteria == 'none':  # 无
        #     ret = True
        # else:
        #     raise Exception(
        #         'Please select criteria in [c1_confi, c2_confi, confi, c1, c2, none]'
        #     )

        # if data_batch['method'] in config.scene_level:
        #     ret = True

        # method = data_batch['method']
        # d_coverage = coverage1 - dict_scene['coverage1']
        # dict_scene['prob'][method] += (1.0 + config.ITER / config.ITERS) * (
        #     d_coverage / 0.03) * (1.0 / dict_scene['prob'][method])

        # DUMPS['prob'][scene] = 1.0 / (0.1 + coverage1)

        # if ret: dict_scene['scene_len'] += 1

        # method = data_batch['method']
        # if args.criteria == 'c1':
        #     d_c1 = max(coverage1 - ref_coverage1, 0)
        #     if d_c1 == 0:
        #         dict_scene['prob'][method] = 1
        #     else:
        #         dict_scene['prob'][method] = np.exp(
        #             np.power(coverage1 * d_c1, 1 / 3))
        # elif args.criteria == 'c2':
        #     d_c2 = max(coverage2 - ref_coverage2, 0)
        #     if d_c2 == 0:
        #         dict_scene['prob'][method] = 1
        #     else:
        #         dict_scene['prob'][method] = np.exp(
        #             np.power(coverage2 * d_c2, 1 / 3))
        # elif args.criteria == 'none':
        #     dict_scene['prob'][method] = 1
        # else:
        #     raise Exception('Please select criteria in [c1, c2, none]')

        # dict_scene['coverage1'] = coverage1
        # dict_scene['coverage2'] = coverage2

        # args.DUMPS['crash'].append(int(not ret))
        # args.DUMPS['crash'][-1] += args.DUMPS['crash'][-2]

        # print(f'in queue : {ret}')

        # d_sec = np.inf
        # for _sg in ref_cluster:
        #     d_sec = min(d_sec, bin(sg_encode ^ _sg).count('1'))

        # d_err_sec = np.inf
        # for _sg in ref_err_cluster:
        #     d_err_sec = min(d_err_sec, bin(err_sg_encode ^ _sg).count('1'))

        # eps = np.finfo(np.float32).eps

        return scene,\
               gt_polygon,\
               sg_encode,\
               err_polygon,\
               err_sg_encode,\
               cal_single_c1(gt_polygon, dict_scene['road_hull']),\
               cal_single_c2(cluster, sg_type),\
               cal_single_c1(err_polygon, dict_scene['road_hull']),\
               cal_single_c2(err_cluster, sg_type),

    return func


def iterate_function(args):

    def func(queue: FuzzQueue, parent_list, mutated_data_batches,
             objective_function):

        # successed = False
        # bug_found = False

        inputs = []

        scene_list = []
        gt_polygon_list = []
        sg_encode_list = []
        err_polygon_list = []
        err_sg_encode_list = []
        results = []

        DUMPS_utlis.updateIter_DUMPS_errorMetric(args.DUMPS)

        for idx in range(len(mutated_data_batches)):
            inputs.append(Seed(parent_list[idx].root_seed, parent_list[idx]))
            res = [0, 0, 0, 0]
            scene, gt_polygon, sg_encode, err_polygon, err_sg_encode, res[
                0], res[1], res[2], res[3] = objective_function(
                    inputs[-1], mutated_data_batches[idx])

            scene_list.append(scene)
            gt_polygon_list.append(gt_polygon)
            sg_encode_list.append(sg_encode)
            err_polygon_list.append(err_polygon)
            err_sg_encode_list.append(err_sg_encode)
            results.append(res)

        for idx in range(len(mutated_data_batches)):
            scene = scene_list[idx]
            gt_polygon = gt_polygon_list[idx]
            sg_encode = sg_encode_list[idx]
            err_polygon = err_polygon_list[idx]
            err_sg_encode = err_sg_encode_list[idx]

            dict_scene = args.DUMPS['scene'][scene]
            sg_type = dict_scene['scene_graph_type']
            dict_cluster = args.DUMPS['cluster'][sg_type]

            dict_scene['gt_polygon'] = shapely.union(dict_scene['gt_polygon'],
                                                     gt_polygon)
            dict_cluster['cluster'].add(sg_encode)
            dict_scene['error_polygon'] = shapely.union(
                dict_scene['error_polygon'], err_polygon)
            dict_cluster['error_cluster'].add(err_sg_encode)

            if sg_encode not in dict_cluster['count_cluster']:
                dict_cluster['count_cluster'][sg_encode] = 0
            dict_cluster['count_cluster'][sg_encode] += 1

            if err_sg_encode not in dict_cluster['count_error_cluster']:
                dict_cluster['count_error_cluster'][err_sg_encode] = 0
            dict_cluster['count_error_cluster'][err_sg_encode] += 1

            # dict_scene['coverage1'] = cal_single_c1(dict_scene['gt_polygon'],
            #                                         dict_scene['road_hull'])
            # dict_cluster['coverage2'] = cal_single_c2(dict_cluster['cluster'],
            #                                           sg_type)
            # dict_scene['error_coverage1'] = cal_single_c1(
            #     dict_scene['error_polygon'], dict_scene['road_hull'])
            # dict_cluster['error_coverage2'] = cal_single_c2(
            #     dict_cluster['error_cluster'], sg_type)

        if len(mutated_data_batches):
            results = np.array(results, dtype=np.float32)

            def strategy_func(x):
                idx = [np.argmax(results[:, x])]
                return idx

            if args.criteria == config.spc:
                idx = strategy_func(0)
            elif args.criteria == config.sec:
                idx = strategy_func(1)
            elif args.criteria == config.error_spc:
                idx = strategy_func(2)
            elif args.criteria == config.error_sec:
                idx = strategy_func(3)
            elif args.criteria == config.none:
                idx = []
            elif args.criteria == config.ldfuzz:
                idx = strategy_func(0) + strategy_func(1)
            elif args.criteria == config.error_mixed:
                idx = strategy_func(2) + strategy_func(3)
            else:
                raise Exception()

            print(f'Select #{idx}')
            for _idx in idx:

                # scene = scene_list[_idx]
                # gt_polygon = gt_polygon_list[_idx]
                # sg_encode = sg_encode_list[_idx]
                # err_polygon = err_polygon_list[_idx]
                # err_sg_encode = err_sg_encode_list[_idx]
                # iou = iou_list[_idx]

                # dict_scene = args.DUMPS['scene'][scene]
                # sg_type = dict_scene['scene_graph_type']
                # dict_cluster = args.DUMPS['cluster'][sg_type]

                # dict_scene['gt_polygon'] = shapely.union(
                #     dict_scene['gt_polygon'], gt_polygon)
                # dict_scene['cluster'].add(sg_encode)
                # dict_scene['error_polygon'] = shapely.union(
                #     dict_scene['error_polygon'], err_polygon)
                # dict_scene['error_cluster'].add(err_sg_encode)

                # dict_scene['coverage1'] = cal_single_c1(
                #     dict_scene['gt_polygon'], dict_scene['road_hull'])
                # dict_scene['coverage2'] = cal_single_c2(
                #     dict_scene['cluster'], sg_type)
                # dict_scene['error_coverage1'] = cal_single_c1(
                #     dict_scene['error_polygon'], dict_scene['road_hull'])
                # dict_scene['error_coverage2'] = cal_single_c2(
                #     dict_scene['error_cluster'], sg_type)

                data_batch = mutated_data_batches[_idx]

                # pd_boxes = data_batch['pred_boxes']
                # ground_truth = data_batch['selected_gt_boxes']
                method = data_batch['method']

                # tp = np.sum(iou >= 0.5)
                # fp = len(pd_boxes) - tp
                # fn = len(ground_truth) - tp
                # data_batch['iou_tp'] = tp

                # args.DUMPS['fp'][method][-1] += fp

                # args.DUMPS['fn'][method][-1] += fn

                # args.DUMPS['fn1'][method][-1] += np.sum(
                #     (data_batch['is_fn'] == False) * (iou < 0.5))
                # data_batch['is_fn'] += iou < 0.5

                # args.DUMPS['fn2'][method][-1] += np.sum(
                #     (data_batch['is_fn2'] == False) * (iou < 0.5))
                # data_batch['is_fn2'] += iou < 0.5

                args.DUMPS['inqueue_method_times'][method] += 1
                queue.save_if_interesting(inputs[_idx], data_batch, False)

            # if not results:
            #     queue.save_if_interesting(input, mutated_data_batches[idx], True)
            #     bug_found = True
            # else:
            #     queue.save_if_interesting(input, mutated_data_batches[idx], False)
            #     successed = True

            # return bug_found, successed

    return func


#############################
# fetch_function
# 返回输出的最后一层一定要是输出层,这样才有label
def fetch_function(model, loader, input_batches, is_eval):
    input_batches = copy.deepcopy(input_batches)
    if input_batches[0]['weather_type'] in WEATHER_METHOD:
        input_batches = WEATHER_METHOD[input_batches[0]['weather_type']](
            input_batches[0], input_batches[0]['w_c'])

    loader.dataset.kitti_infos = input_batches

    batches = list(loader)[0]

    for __tmp__ in batches:
        if not isinstance(batches[__tmp__], torch.Tensor):
            if __tmp__ == 'ref_points': continue
            elif __tmp__ == 'road_hull': continue
            elif __tmp__ == 'road_pc': continue
            elif __tmp__ == 'road_labels': continue
            elif __tmp__ == 'non_road_pc': continue
            elif __tmp__ == 'mtest_calib': continue
            try:
                batches[__tmp__] = torch.tensor(data=batches[__tmp__],
                                                device=config.TORCH_DEVICE)
            except:
                None

    if config.TORCH_DEVICE is 'cpu':
        layer_outputs = model.forward(batches)
    elif config.TORCH_DEVICE is 'cuda':
        layer_outputs = model.cuda().forward(batches)
    else:
        raise Exception('Please select TORCH_DEVICE in [cpu, cuda]')

    # SAVE GPU MEMORY
    del batches
    del input_batches
    torch.cuda.empty_cache()

    pred_boxes = []
    pred_scores = []
    pred_labels = []
    for __i__ in range(len(layer_outputs[0])):
        pred_boxes.append(
            layer_outputs[0][__i__]['pred_boxes'].detach().cpu().numpy())
        pred_scores.append(
            layer_outputs[0][__i__]['pred_scores'].detach().cpu().numpy())
        pred_labels.append(
            layer_outputs[0][__i__]['pred_labels'].detach().cpu().numpy())

    del layer_outputs
    torch.cuda.empty_cache()

    if is_eval: return pred_scores, pred_boxes, pred_labels
    else: return pred_scores, pred_boxes


def quantize_fetch_function(handler, input_batches, preprocess, models):
    _, img_batches, _, _, _ = input_batches
    if len(img_batches) == 0:
        return None, None
    preprocessed = preprocess(img_batches)

    layer_outputs = handler.predict(preprocessed)
    results = np.expand_dims(np.argmax(layer_outputs[-1], axis=1), axis=0)
    for m in models:
        r1 = np.expand_dims(np.argmax(m.predict(preprocessed), axis=1), axis=0)
        results = np.append(results, r1, axis=0)
    # Return the prediction outputs of all models
    return layer_outputs, results


def build_fetch_function(model, loader, is_eval=False):

    def func(input_batches):
        # """The fetch function."""
        # if models is not None:
        #     return quantize_fetch_function(handler, input_batches, preprocess,
        #                                    models)
        # else:
        return fetch_function(model, loader, input_batches, is_eval)

    return func


def build_frd_function(model):

    def func(input_batches):
        return model.infer(input_batches)

    return func
