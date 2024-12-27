from .config import object_level, scene_level
from .coverage_utils import get_scene_graph_type, get_scene_graph_encode, get_gt_polygon, get_coverage1_area, get_coverage1_DUMPSver, get_coverage2_count, get_coverage2_DUMPSver
from shapely.geometry import Polygon

error_metric = [
    'fp', 'fn', 'fn1', 'fn2', 'fn3', 'obj_missing', 'false_det', 'fail_loc'
]


def init_DUMPS(DUMPS):
    # DUMPS.clear()
    DUMPS['scene'] = {}
    DUMPS['cluster'] = {}
    DUMPS['fp'] = {i: [0] for i in object_level + scene_level}
    DUMPS['fn'] = {i: [0] for i in object_level + scene_level}
    DUMPS['fn1'] = {i: [0] for i in object_level + scene_level}
    DUMPS['fn2'] = {i: [0] for i in object_level + scene_level}
    DUMPS['fn3'] = {i: [0] for i in object_level + scene_level}
    DUMPS['obj_missing'] = {i: [0] for i in object_level + scene_level}
    DUMPS['false_det'] = {i: [0] for i in object_level + scene_level}
    DUMPS['fail_loc'] = {i: [0] for i in object_level + scene_level}
    DUMPS['crash'] = {i: [0] for i in object_level + scene_level}
    DUMPS['error_detail'] = []
    DUMPS['fa'] = {}
    DUMPS['prob'] = {}
    DUMPS['inqueue_method_times'] = {i: 0 for i in object_level + scene_level}
    DUMPS['coverage1_list'] = []
    DUMPS['coverage2_list'] = []
    DUMPS['error_coverage1_list'] = []
    DUMPS['error_coverage2_list'] = []
    DUMPS['coverage1_area_list'] = []
    DUMPS['error_coverage1_area_list'] = []
    DUMPS['coverage2_count_list'] = []
    DUMPS['error_coverage2_count_list'] = []
    DUMPS['frd_limit'] = 0


def updateIter_DUMPS_errorMetric(DUMPS):
    for m in error_metric:
        for _method in DUMPS[m]:
            DUMPS[m][_method].append(DUMPS[m][_method][-1])


def updateBatches_DUMPS(DUMPS, seed_name, batch, metadata_list):
    batch['level'] = 1
    batch['pred_boxes'] = metadata_list
    batch['criteria'] = DUMPS['criteria']

    kitti_idx = batch['point_cloud']['lidar_idx']
    points = batch['points']
    selected_gt_boxes = batch['selected_gt_boxes']
    selected_name = batch['selected_name']
    weather_type = batch['weather_type']

    road_hull = batch['road_hull']
    sg_type = get_scene_graph_type(road_hull)
    batch['scene_graph_type'] = sg_type

    scene = seed_name.split('.')[0]
    batch['scene'] = scene

    sg_encode = get_scene_graph_encode(selected_gt_boxes, selected_name,
                                       weather_type, sg_type)
    gt_polygon = get_gt_polygon(selected_gt_boxes)

    cluster = set([sg_encode])

    dict_scene = DUMPS['scene']
    dict_cluster = DUMPS['cluster']

    dict_scene.update({
        scene: {
            'kitti_idx': kitti_idx,
            'scene_graph_type': sg_type,
            'points': points,
            'gt_polygon': gt_polygon,
            'error_polygon': Polygon(),
            'road_hull': road_hull,
            'cluster': cluster,
            'error_cluster': set({0}),
            'scene_len': 0,
            'method_times': {
                i: 0
                for i in object_level + scene_level
            },
            'total_times': 0
        }
    })

    if sg_type in dict_cluster:
        dict_cluster[sg_type]['cluster'].add(sg_encode)
        dict_cluster[sg_type]['scene_list'].append(scene)
    else:
        dict_cluster[sg_type] = {
            'cluster': set([sg_encode]),
            'error_cluster': set({0}),
            'scene_list': [scene],
            'count_cluster': {},
            'count_error_cluster': {}
        }


def updateCoverage_DUMPS(DUMPS):
    cov1 = get_coverage1_DUMPSver(DUMPS)
    cov2 = get_coverage2_DUMPSver(DUMPS)
    c1_area = get_coverage1_area(DUMPS)
    c2_count = get_coverage2_count(DUMPS)

    err_cov1 = get_coverage1_DUMPSver(DUMPS, True)
    err_cov2 = get_coverage2_DUMPSver(DUMPS, True)
    err_c1_area = get_coverage1_area(DUMPS, True)
    err_c2_count = get_coverage2_count(DUMPS, True)

    DUMPS['coverage1'] = cov1
    DUMPS['coverage2'] = cov2
    DUMPS['error_coverage1'] = err_cov1
    DUMPS['error_coverage2'] = err_cov2

    DUMPS['coverage1_list'].append(cov1)
    DUMPS['coverage2_list'].append(cov2)
    DUMPS['error_coverage1_list'].append(err_cov1)
    DUMPS['error_coverage2_list'].append(err_cov2)

    DUMPS['coverage1_area_list'].append(c1_area)
    DUMPS['error_coverage1_area_list'].append(err_c1_area)
    DUMPS['coverage2_count_list'].append(c2_count)
    DUMPS['error_coverage2_count_list'].append(err_c2_count)
