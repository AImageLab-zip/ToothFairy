import os

dataset_root = os.environ.get(
    'BITE2TEXT_PTV3_DATA_ROOT',
    'data/bite2text_ptv3_surface32k_v3_official_12head_full867',
)

weight = 'exp/dental/bite2text_ptv3_v3_official_12head_full867_stage1_frozen_seed20260815/model/model_last.pth'

resume = False

evaluate = False

test_only = False

seed = 20260815

save_path = 'exp/dental/bite2text_ptv3_v3_official_12head_full867_stage2_joint_seed20260815'

num_worker = 4

batch_size = 8

gradient_accumulation_steps = 1

batch_size_val = 8

batch_size_test = 8

epoch = 47

eval_epoch = 47

clip_grad = 1.0

sync_bn = False

enable_amp = True

amp_dtype = 'float16'

empty_cache = False

empty_cache_per_epoch = False

find_unused_parameters = False

enable_wandb = False

wandb_project = 'bite2text-ptv3'

wandb_key = None

mix_prob = 0

selection_metric = 'f1'

param_dicts = [{'keyword': 'backbone', 'lr': 1e-05}]

hooks = [{'type': 'CheckpointLoader', 'keywords': 'module.', 'replacement': 'module.', 'strict': True},
 {'type': 'IterationTimer'},
 {'type': 'InformationWriter'},
 {'type': 'MultiClsEvaluator'},
 {'type': 'CheckpointSaver', 'save_freq': None}]

train = {'type': 'DefaultTrainer'}

test = {'type': 'MultiClsTester', 'verbose': True}

data = {'names': ['right_molar_relation',
           'right_canine_relation',
           'left_molar_relation',
           'left_canine_relation',
           'overjet',
           'vertical_relation',
           'midline_relation',
           'crossbite',
           'upper_crowding',
           'lower_crowding',
           'curve_spee',
           'curve_wilson'],
 'train': {'type': 'Bite2TextDataset',
           'split': 'train',
           'data_root': dataset_root,
           'transform': [{'type': 'NormalizeCoord'},
                         {'type': 'RandomScale', 'scale': [0.95, 1.05]},
                         {'type': 'RandomShift', 'shift': ((-0.02, 0.02), (-0.02, 0.02), (-0.02, 0.02))},
                         {'type': 'RandomRotate',
                          'angle': [-0.1, 0.1],
                          'axis': 'z',
                          'center': [0, 0, 0],
                          'p': 0.5},
                         {'type': 'RandomDropout', 'dropout_ratio': 0.35, 'dropout_application_ratio': 0.5},
                         {'type': 'GridSample',
                          'grid_size': 0.01,
                          'hash_type': 'fnv',
                          'mode': 'train',
                          'return_grid_coord': True},
                         {'type': 'ShufflePoint'},
                         {'type': 'ToTensor'},
                         {'type': 'Collect',
                          'keys': ('coord',
                                   'grid_coord',
                                   'label_0',
                                   'label_1',
                                   'label_2',
                                   'label_3',
                                   'label_4',
                                   'label_5',
                                   'label_6',
                                   'label_7',
                                   'label_8',
                                   'label_9',
                                   'label_10',
                                   'label_11'),
                          'feat_keys': ['coord', 'point_label_onehot']}],
           'loop': 1,
           'max_samples': 0},
 'val': {'type': 'Bite2TextDataset',
         'split': 'val',
         'data_root': dataset_root,
         'test_mode': False,
         'transform': [{'type': 'NormalizeCoord'},
                       {'type': 'GridSample',
                        'grid_size': 0.01,
                        'hash_type': 'fnv',
                        'mode': 'train',
                        'return_grid_coord': True},
                       {'type': 'ToTensor'},
                       {'type': 'Collect',
                        'keys': ('coord',
                                 'grid_coord',
                                 'name',
                                 'label_0',
                                 'label_1',
                                 'label_2',
                                 'label_3',
                                 'label_4',
                                 'label_5',
                                 'label_6',
                                 'label_7',
                                 'label_8',
                                 'label_9',
                                 'label_10',
                                 'label_11'),
                        'feat_keys': ['coord', 'point_label_onehot']}],
         'max_samples': 0},
 'test': {'type': 'Bite2TextDataset',
          'split': 'test',
          'data_root': dataset_root,
          'test_mode': False,
          'transform': [{'type': 'NormalizeCoord'},
                        {'type': 'GridSample',
                         'grid_size': 0.01,
                         'hash_type': 'fnv',
                         'mode': 'train',
                         'return_grid_coord': True},
                        {'type': 'ToTensor'},
                        {'type': 'Collect',
                         'keys': ('coord',
                                  'grid_coord',
                                  'name',
                                  'label_0',
                                  'label_1',
                                  'label_2',
                                  'label_3',
                                  'label_4',
                                  'label_5',
                                  'label_6',
                                  'label_7',
                                  'label_8',
                                  'label_9',
                                  'label_10',
                                  'label_11'),
                         'feat_keys': ['coord', 'point_label_onehot']}],
          'max_samples': 0}}

model = {'type': 'MultiTaskClassifier',
 'num_classes_list': [6, 6, 6, 6, 5, 5, 3, 4, 6, 6, 2, 2],
 'class_weights': [[0.36848171, 0.59236656, 0.62109225, 2.03976701, 1.02911341, 1.34917906],
                   [0.43017825, 0.6420394, 0.64986944, 1.59675409, 1.2368404, 1.44431842],
                   [0.35256284, 0.53174207, 0.6151305, 2.26848918, 0.87639383, 1.35568158],
                   [0.37182609, 0.4850046, 0.58709157, 2.34836627, 1.01687242, 1.19083905],
                   [0.42509177, 0.40827843, 1.1279613, 2.0386685, 1.0],
                   [0.86589116, 0.94853624, 1.39966779, 0.72046499, 1.06543982],
                   [1.04703378, 1.22496241, 0.72800381],
                   [0.45252091, 2.00679972, 0.85570228, 0.68497709],
                   [0.46019847, 0.29905373, 1.84079389, 0.71895285, 1.47591858, 1.20508247],
                   [0.84114148, 0.45204596, 1.50630017, 0.84570055, 1.15881067, 1.19600118],
                   [1.05217787, 0.94782213],
                   [0.99293528, 1.00706472]],
 'loss_type': 'ce',
 'backbone_embed_dim': 128,
 'freeze_backbone': False,
 'backbone': {'type': 'PT-v3m1',
              'in_channels': 9,
              'enc_channels': (16, 32, 48, 64, 128),
              'enc_num_head': (1, 2, 3, 4, 8),
              'dec_channels': (32, 32, 64, 96),
              'dec_num_head': (2, 2, 4, 6),
              'enable_flash': False,
              'enc_mode': True}}

optimizer = {'type': 'AdamW', 'lr': 0.0001, 'weight_decay': 0.01}

scheduler = {'type': 'CosineAnnealingLR', 'total_steps': 47}
