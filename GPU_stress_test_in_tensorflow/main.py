import tensorflow as tf
from tensorflow.python.client import device_lib
from model.model import Tyan_korea
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import argparse

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

test_input_resolution=224
end_count=50

def parse_args():
        parser = argparse.ArgumentParser(description='get_vram')
        parser.add_argument('--vram', type=int, default='8',help='gpu_vram')
        args = parser.parse_args()
        return args

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def main(GPU_num):
    args=parse_args()
    vram=args.vram
    batch_size=vram
    model=Tyan_korea(GPU_num,batch_size)

    input_place_holder=tf.placeholder(shape=(batch_size,test_input_resolution,test_input_resolution,3),dtype=tf.float32)


    forwarding_result=model.test(intput=input_place_holder)

    fake_image=np.zeros(shape=(batch_size,test_input_resolution,test_input_resolution,3),dtype=np.float32)

    test_count=0

    config = tf.ConfigProto(
        # device_count={'GPU': 1},
        log_device_placement=False,
        allow_soft_placement=True

    )

    with tf.Session(config=config) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        init = tf.global_variables_initializer()
        sess.run(init)

        while True:


            forwarding=sess.run(forwarding_result,feed_dict={input_place_holder : fake_image})
            test_count+=1

            if test_count%10 ==0:
                print("%d / %d"%(test_count,end_count))


            if test_count== end_count:
                break
        coord.request_stop()

if __name__ == '__main__':

    GPU_num=len(get_available_gpus())
    main(GPU_num)
