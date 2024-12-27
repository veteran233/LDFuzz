mnist = "mnist"
fashion = "fashion"
cifar10 = "cifar"
svhn = "svhn"
iamge_net = "iamgenet"
kitti = "kitti"

LeNet5 = "LeNet5"
LeNet1 = "LeNet1"
resNet20 = "resNet20"
vgg16 = "vgg16"

pointpillar = "pointpillar"
pv_rcnn = "pv_rcnn"
second = "second"
pointrcnn = "pointrcnn"

spc = 'spc'
sec = 'sec'
ldfuzz = 'ldfuzz'
error_spc = 'error_spc'
error_sec = 'error_sec'
error_mixed = 'error_mixed'
none = 'none'

# model_weight_path = {
#     'vgg16': "__profile/cifar10/models/vgg16.h5",
#     'resnet20': "__profile/cifar10/models/resnet.h5",
#     'LeNet1': "__profile/mnist/models/LeNet1.hdf5",
#     'lenet4': "__profile/mnist/models/lenet4.h5",
#     'LeNet5': "__profile/mnist/models/LeNet5.hdf5"
# }

model_data = {
    # mnist: [LeNet5, LeNet1],
    # fashion: [LeNet1, resNet20],
    # cifar10: [vgg16, resNet20],
    # svhn: [LeNet5, vgg16]
    kitti: [pointpillar, pv_rcnn, second, pointrcnn]
}

TORCH_DEVICE = 'cuda'

# clear && conda activate r_deephunter && python rq1step1.py
# clear && conda activate r_deephunter && python run_fuzzer.py -m pointpillar,pv_rcnn -gc ldfuzz,none -s new -mi 10

# python run_fuzzer.py

object_level = ['translocate', 'rotation', 'scale', 'insert']
scene_level = ['rain', 'snow', 'fog']

# criteria_arr = [ldfuzz, none]
# select_arr = ['new']

# ---------------------------
'''
|----|----| <-- 70m
| 6  | 5  |
|----|----| <-- 40m
| 4  | 3  |
|----|----| <-- 20m
| 2  | 1  |
|----|----| <-- 0m
^    ^    ^
|    |    |
40m  0m  -40m
'''
# ---------------------------
scene_graph_width_list = [(-40, 0), (0, 40)]
scene_graph_length_list = [(0, 20), (20, 40), (40, 70)]


def get_model_weight_path(data_name, model_name):
    if model_name == resNet20 and data_name == cifar10:
        suffix = "h5"
    else:
        suffix = "hdf5"
    return "__profile/{}/models/{}.{}".format(data_name, model_name, suffix)


def get_seed_path(data_name, model_name):
    seed_path = "./__test_seeds/{}_{}".format(data_name, model_name)
    return seed_path


def get_output_base_path(data_name, model_name, i=None):
    if i is None:
        output_base_path = "__out/{}_{}".format(data_name, model_name)
    else:
        output_base_path = "__out{}/{}_{}".format(i, data_name, model_name)
    return output_base_path


def get_output_path(
        data_name,
        model_name,
        select,
        criteria,
        # mutate_strategy,
        num=0,
        i=None):
    output_base_path = get_output_base_path(data_name, model_name, i=i)
    output_dir = "{}/{}/{}/{}".format(output_base_path, select, criteria, num)
    return output_dir


def get_model_profile_base_path(data_name, model_name):
    return "./__profile/{}/profiling/{}".format(data_name, model_name)


def get_model_profile_path(data_name, model_name, pickle_name="0_60000"):
    base_path = get_model_profile_base_path(data_name, model_name)
    return "{}/{}.pickle".format(base_path, pickle_name)


# shape_dic = {
#     'vgg16': (32, 32, 3),
#     'resnet20': (32, 32, 3),
#     'LeNet1': (28, 28, 1),
#     'lenet4': (28, 28, 1),
#     'LeNet5': (28, 28, 1),
#     'mobilenet': (224, 224, 3),
#     'vgg19': (224, 224, 3),
#     'resnet50': (224, 224, 3)
# }
metrics_para = {
    'kmnc': 1000,
    'bknc': 10,
    'tknc': 10,
    'nbc': 10,
    'newnc': 10,
    'nc': 0.75,
    'fann': 1.0,
    'snac': 10,
    'space': 4,
    'fake': 0,
    "lsc": 1000
}
execlude_layer_dic = {
    vgg16: [
        'input', 'flatten', 'padding', 'activation', 'batch', 'dropout', 'bn',
        'reshape', 'relu', 'pool', 'concat', 'softmax', 'fc'
    ],
    resNet20: [
        'input', 'flatten', 'padding', 'activation', 'batch', 'dropout', 'bn',
        'reshape', 'relu', 'pool', 'concat', 'add', 'res4', 'res5'
    ],
    LeNet1: ['input', 'flatten', 'activation', 'batch', 'dropout'],
    LeNet5: ['input', 'flatten', 'activation', 'batch', 'dropout'],
    'lenet4': ['input', 'flatten', 'activation', 'batch', 'dropout'],
    'mobilenet': [
        'input', 'flatten', 'padding', 'activation', 'batch', 'dropout', 'bn',
        'reshape', 'relu', 'pool', 'concat', 'softmax', 'fc'
    ],
    'vgg19': [
        'input', 'flatten', 'padding', 'activation', 'batch', 'dropout', 'bn',
        'reshape', 'relu', 'pool', 'concat', 'softmax', 'fc'
    ],
    'resnet50': [
        'input', 'flatten', 'padding', 'activation', 'batch', 'dropout', 'bn',
        'reshape', 'relu', 'pool', 'concat', 'add', 'res4', 'res5'
    ],
    pointpillar: [],
    pv_rcnn: [],
    second: []
}
