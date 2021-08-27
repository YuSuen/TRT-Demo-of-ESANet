# _*_ coding: utf-8 _*_
"""
@Time   : 2021/8/9
@Author : YuSuen
"""

import os
import rospy
import message_filters
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError

import argparse
import cv2
import mock  # pip install mock
import numpy as np
import torch

from src.args import ArgumentParserRGBDSegmentation
from src.models.model_utils import SqueezeAndExcitationTensorRT
from src.prepare_data import prepare_data

with mock.patch('src.models.model_utils.SqueezeAndExcitation',
                SqueezeAndExcitationTensorRT):
    from src.build_model import build_model


def _parse_args():
    parser = ArgumentParserRGBDSegmentation(
        description='Efficient RGBD Indoor Sematic Segmentation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.set_common_args()
    parser.add_argument('--model', type=str, default='onnx',
                        choices=['own', 'onnx'],
                        help='The model for which the inference time will be'
                             'measured.')
    parser.add_argument('--model_onnx_filepath', type=str, default='./onnx_models/model.onnx',
                        help="Path to ONNX model file when --model is 'onnx'")

    # runs
    parser.add_argument('--n_runs', type=int, default=100,
                        help='For how many runs the inference time will be '
                             'measured')
    parser.add_argument('--n_runs_warmup', type=int, default=10,
                        help='How many forward paths trough the model before'
                             'the inference time measurements starts. This is '
                             'necessary as the first runs are slower.')
    # timings
    parser.add_argument('--no_time_pytorch', dest='time_pytorch',
                        action='store_false', default=False,
                        help='Set this if you do not want to measure the'
                             'pytorch times.')
    parser.add_argument('--no_time_tensorrt', dest='time_tensorrt',
                        action='store_false', default=True,
                        help='Set this if you do not want to measure the '
                             'tensorrt times.')
    parser.add_argument('--no_time_onnxruntime', dest='time_onnxruntime',
                        action='store_false', default=False,
                        help='Set this if you do not want to measure the '
                             'onnxruntime times.')
    # plots / export
    parser.add_argument('--plot_timing', default=False, action='store_true',
                        help='Wether to plot the inference time for each'
                             'forward pass')
    parser.add_argument('--plot_outputs', default=True, action='store_true',
                        help='Wether to plot the colored segmentation output'
                             'of the model')
    parser.add_argument('--export_outputs', default=True, action='store_true',
                        help='Whether to export the colored segmentation output'
                             'of the model to png')

    # tensorrt
    parser.add_argument('--trt_workspace', type=int, default=2 << 30,
                        help='default is 2GB')
    parser.add_argument('--trt_floatx', type=int, default=32, choices=[16, 32],
                        help='Whether to measure tensorrt timings with float16'
                             'or float32.')
    parser.add_argument('--trt_batchsize', type=int, default=1)
    parser.add_argument('--trt_onnx_opset_version', type=int, default=10,
                        help='different versions lead to different results but'
                             'not all versions are supported for the following'
                             'tensorrt conversion.')
    parser.add_argument('--trt_dont_force_rebuild', dest='trt_force_rebuild',
                        default=False, action='store_false',
                        help='Possibly already existing trt engine file will '
                             'be reused when providing this argument.')
    parser.add_argument('--onnxruntime_onnx_opset_version', type=int,
                        default=11,
                        help='opset 10 leads to different results compared to'
                             'PyTorch')
    # see: https://github.com/microsoft/onnxruntime/blob/master/docs/execution_providers/TensorRT-ExecutionProvider.md
    parser.add_argument('--onnxruntime_trt_max_partition_iterations', type=str,
                        default='15',
                        help='maximum number of iterations allowed in model '
                             'partitioning for TensorRT')
    parser.add_argument('--depth_scale', type=float,
                        default=0.1,
                        help='Additional depth scaling factor to apply.')

    args = parser.parse_args()
    args.pretrained_on_imagenet = False
    return args


# TensorRT
class ESATRT(object):
    def __init__(self, engine_file_path):
        # Create a Context on this device,
        self.cfx = cuda.Device(0).make_context()
        stream = cuda.Stream()

        # Deserialize the engine from file
        print(f"Loading engine: {engine_file_path}")
        with open(engine_file_path, "rb") as f, \
                trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())

        context = engine.create_execution_context()

        in_cpu = []
        in_gpu = []
        for i in range(engine.num_bindings - 1):
            shape = trt.volume(engine.get_binding_shape(i))
            dtype = trt.nptype(engine.get_binding_dtype(i))

            in_cpu.append(cuda.pagelocked_empty(shape, dtype))
            in_gpu.append(cuda.mem_alloc(in_cpu[-1].nbytes))

        # output binding
        shape = trt.volume(engine.get_binding_shape(engine.num_bindings - 1))
        dtype = trt.nptype(engine.get_binding_dtype(engine.num_bindings - 1))
        out_cpu = cuda.pagelocked_empty(shape, dtype)
        out_gpu = cuda.mem_alloc(out_cpu.nbytes)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.in_cpu = in_cpu
        self.out_cpu = out_cpu
        self.in_gpu = in_gpu
        self.out_gpu = out_gpu

    def infer(self, inputs):
        threading.Thread.__init__(self)
        # Make self the active context, pushing it on top of the context stack.
        self.cfx.push()

        # self.stream = stream
        context = self.context
        # self.engine = engine
        # self.in_cpu = in_cpu
        out_cpu = self.out_cpu
        in_gpu = self.in_gpu
        out_gpu = self.out_gpu

        pointers = [int(in_) for in_ in in_gpu] + [int(out_gpu)]
        outs = []
        for i in range(len(inputs[0])):
            # copy to gpu (do not use for loop)
            # cuda.memcpy_htod(in_gpu[0], inputs[0][i].numpy())
            cuda.memcpy_htod(in_gpu[0], np.ascontiguousarray(inputs[0][i].numpy()))
            if len(inputs) == 2:
                cuda.memcpy_htod(in_gpu[1], inputs[1][i].numpy())

            # model forward pass
            context.execute(1, pointers)

            # copy back to cpu
            cuda.memcpy_dtoh(out_cpu, out_gpu)

            outs.append(out_cpu.copy())

        return outs


import threading


class myThread(threading.Thread):
    def __init__(self, func, args):
        threading.Thread.__init__(self)
        self.func = func
        self.args = args

    def run(self):
        self.func(*self.args)


# ROS
class ESAROS():
    def __init__(self):
        self.con_flag = True
        self._cv_bridge = CvBridge()
        self.flag = 0

        self.dataset, self.preprocessor = prepare_data(args, with_input_orig=True)
        self.n_classes = self.dataset.n_classes_without_void

        engine_file_path = '/home/siu/PycharmProject/ESANet/onnx_models/model.trt'
        self.easnet_wrapper = ESATRT(engine_file_path)

        rgb = message_filters.Subscriber('/zed2/zed_node/rgb/image_rect_color/compressed', CompressedImage,
                                         queue_size=10,
                                         buff_size=2 ** 24)
        depth = message_filters.Subscriber('/zed2/zed_node/depth/depth_registered', Image, queue_size=10,
                                           buff_size=2 ** 24)

        ts = message_filters.ApproximateTimeSynchronizer([rgb, depth], 10, 0.1, allow_headerless=True)

        ts.registerCallback(self.callback)

    def callback(self, rgb, depth):
        rgb_image = cv2.imdecode(np.fromstring(rgb.data, dtype=np.int8), cv2.IMREAD_COLOR)
        rgb_image = rgb_image.astype('uint8')

        depth_image = self._cv_bridge.imgmsg_to_cv2(depth, '32FC1')
        depth_image = np.array(depth_image, dtype=np.float)
        depth_image = depth_image * 1000.0
        depth_image = np.round(depth_image).astype(np.uint16)

        img_rgb = rgb_image
        img_depth = depth_image
        img_depth = img_depth.astype('float32') * args.depth_scale
        h, w, _ = img_rgb.shape

        # preprocess sample
        sample = self.preprocessor({'image': img_rgb, 'depth': img_depth})
        rgb_images = (sample['image'][None])
        depth_images = (sample['depth'][None])

        # n_classes_without_void = self.dataset.n_classes_without_void

        inputs = (rgb_images, depth_images)

        out_tensorrt = self.easnet_wrapper.infer(inputs)
        out = out_tensorrt[0].reshape(-1, args.height, args.width)

        argmax_tensorrt = np.argmax(out, axis=0).astype(np.uint8) + 1
        print(np.unique(argmax_tensorrt))

        colored = self.dataset.color_label(argmax_tensorrt)

        colored = cv2.resize(colored, (w, h), interpolation=cv2.INTER_CUBIC)

        # show
        cv2.imshow('tensorrt', colored)
        cv2.imshow('depth', depth_image)
        cv2.imshow('rgb', img_rgb)
        cv2.waitKey(1)


if __name__ == '__main__':
    args = _parse_args()
    print(f"args: {vars(args)}")
    print('PyTorch version:', torch.__version__)

    if args.time_tensorrt:
        import tensorrt as trt
        import pycuda.autoinit
        import pycuda.driver as cuda

        print('TensorRT version:', trt.__version__)

    if args.time_onnxruntime:
        import onnxruntime

        onnxruntime_profile_execution = True

        # see: https://github.com/microsoft/onnxruntime/blob/master/docs/execution_providers/TensorRT-ExecutionProvider.md
        os.environ['ORT_TENSORRT_MAX_WORKSPACE_SIZE'] = str(2 << 30)
        os.environ['ORT_TENSORRT_MIN_SUBGRAPH_SIZE'] = '1'  # 5
        # note, 1 does not raise an error if not available but enabled
        os.environ['ORT_TENSORRT_FP16_ENABLE'] = '0'  # 1
        os.environ['ORT_TENSORRT_MAX_PARTITION_ITERATIONS'] = \
            args.onnxruntime_trt_max_partition_iterations

        print('ONNXRuntime version:', onnxruntime.__version__)
        print('ONNXRuntime available providers:',
              onnxruntime.get_available_providers())

    gpu_devices = torch.cuda.device_count()

    # prepare inputs ----------------------------------------------------------
    label_downsampling_rates = []
    results_dir = os.path.join(os.path.dirname(__file__),
                               f'inference_results_{args.upsampling}',
                               args.dataset)
    os.makedirs(results_dir, exist_ok=True)
    args.batch_size = 1
    args.batch_size_valid = 1

    rospy.init_node('esanet')
    ESANet = ESAROS()
    rospy.spin()
