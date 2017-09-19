import os

import gym
import tensorflow as tf

from . import option
from .model import MLPModel, CNNModel
from .ppo import PPO
from .worker import TrainWorker


def mlp_model_func(ob_space, ac_space): return MLPModel(ob_space, ac_space, ob_filter=True, gaussian_fixed_var=True)

def cnn_model_func(ob_space, ac_space): return CNNModel(ob_space, ac_space, kind='large')

def main():
    # parse args
    args = option.args

    # worker device
    if args.backend == 'cpu':
        args.worker_device = "/cpu:0"
    else:
        gpu_id = args.index % args.gpu_count
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        args.worker_device = "/gpu:0"


    # start session
    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        gpu_options=tf.GPUOptions(allow_growth=True)
    )
    sess = tf.Session(config=config)
    sess.__enter__()

    # create env
    env = gym.make(args.env).unwrapped

    # create ppo
    with tf.device(args.worker_device):
        ppo = PPO(env.observation_space, env.action_space, cnn_model_func, clip_param=0.2, entcoeff=0.01)

    # create worker
    if args.mode == 'train':
        worker = TrainWorker(env, ppo, args.render,
                    train_data_size=256, optimize_size=64, optimize_epochs=4,
                    gamma=0.99, lambda_=0.95, max_steps=1e6)

    else:
        pass


    # start worker
    worker()


if __name__ == '__main__':
    main()