import os
import sagemaker
from sagemaker import get_execution_role

sagemaker_session = sagemaker.Session()

role = get_execution_role()
region = sagemaker_session.boto_session.region_name

image = "763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.4.1-gpu-py37-cu110-ubuntu18.04"
s3_output = "s3://sagemaker-us-east-1-astra-face-recognition/output"

# MODEL_TYPE = "resnet50"  # se_resnet50 | resnet50 | mobilenetv3l | mobilenetv3s | mobilenetv2
MODEL_TYPE = "res50v2"  # res50v2 | mobilenetv3l | efficientnet_b4 | efficientnet_b3 | efficientnet_b2

PRETRAIN = "20210608_t2_epoch_5.h5"
# PRETRAIN = "None"
EPOCH = 5
LR = 0.01
DEBUG = False
VERBOSE = 2
BRACNCH = 'exp'
BUFFER_SIZE = 200000
FREEZE_LAYERS = 0
# INSTANCE_TYPE = 'ml.p3.8xlarge'
INSTANCE_TYPE = 'ml.g4dn.16xlarge'
BATCH_SIZE = 32

if DEBUG:
    hyperparameters = {"task_name": "fr-train",
                          "image_size" : 112,
                          "batch_size" : 2,
                          "epoch": EPOCH,
                          "freq_factor_by_number_of_epoch": 1,
                          "pretrained":PRETRAIN,
                          "num_of_class":10,
                          "train_image_count":22,
                          "valid_image_count":30,
                          "modeltype": MODEL_TYPE,
                          "verbose":VERBOSE,
                          "max_ckpt":1,
                          "lr":LR,
                          "freeze_layers":FREEZE_LAYERS,
                          "buffer_size":10}
else:
    hyperparameters = {"task_name": "fr-train",
                          "image_size" : 112,
                          "batch_size" : BATCH_SIZE,
                          "epoch": EPOCH,
                          "freq_factor_by_number_of_epoch": 1,
                          "pretrained":PRETRAIN,
                          "num_of_class":93979, # 20000
                          "train_image_count":2033692, # 663118
                          "valid_image_count":796456, # 60000
                          "modeltype": MODEL_TYPE,
                          "verbose":VERBOSE,
                          "max_ckpt":1,
                          "lr":LR,
                          "freeze_layers":FREEZE_LAYERS,
                          "buffer_size":BUFFER_SIZE}




metric_definition = [
    {'Name': 'train:loss', 'Regex': '.*loss: ([0-9\\.]+) - acc: [0-9\\.]+.*'},
    {'Name': 'train:accuracy', 'Regex': '.*loss: [0-9\\.]+ - acc: ([0-9\\.]+).*'},
    {'Name': 'sec/steps', 'Regex': '.* - \d+s (\d+)[mu]s/step - loss: [0-9\\.]+ - acc: [0-9\\.]+ - val_loss: [0-9\\.]+ - val_acc: [0-9\\.]+'}
]


from sagemaker.tensorflow import TensorFlow
git_config = {'repo': 'https://github.com/jason9075/tf2_arcface', 'branch': BRACNCH}
MAX_TRAINING_TIME = 24 * 60 * 60 * 5 # 5天

tf_estimator = TensorFlow(entry_point="test_train_mirror.py",
                             image_uri=image,
                             source_dir='./',
                             git_config=git_config,
                             role=role,
                             instance_count=1,
                             max_run=MAX_TRAINING_TIME,
                             instance_type=INSTANCE_TYPE,
                             output_path=s3_output,
                             sagemaker_session=sagemaker_session,
                             hyperparameters=hyperparameters,
                             metric_definitions=metric_definition,
                             input_mode='Pipe')

# training_data_uri = 's3://astra-face-recognition-dataset/divide/'
if DEBUG:
    training_data_uri = {'train':'s3://sagemaker-us-east-1-astra-face-recognition/tfrecord/divide.tfrecord', 
                        'valid':'s3://sagemaker-us-east-1-astra-face-recognition/tfrecord/divide.tfrecord'}
else:
    training_data_uri = {'train':'s3://sagemaker-us-east-1-astra-face-recognition/tfrecord/93979_train.tfrecord',
                        'valid':'s3://sagemaker-us-east-1-astra-face-recognition/tfrecord/93979_valid.tfrecord'}
#     training_data_uri = {'train':'s3://sagemaker-us-east-1-astra-face-recognition/tfrecord/20000_dataset_cc.tfrecord',
#                         'valid':'s3://sagemaker-us-east-1-astra-face-recognition/tfrecord/20000_dataset_aug.tfrecord'}

tf_estimator.fit(training_data_uri)
