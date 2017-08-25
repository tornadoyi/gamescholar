import os
import sys
import multiprocessing
import time
import copy
import importlib.util
import platform
import logging
from easydict import EasyDict as edict
import tensorflow as tf
import option


def listen_shutdown(callback=None):
    sysstr = platform.system()
    if sysstr == "Windows": return

    import sys, signal
    def shutdown(signal, frame):
        logging.info('process {} received signal {}: exiting'.format(os.getpid(), signal))
        if callback is not None: callback()
        sys.exit(128 + signal)

    signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)


def get_gpu_count():
    result = os.popen('python -c \"{} {} {}\"'.format(
        'from tensorflow.python.client import device_lib;',
        'gpus=[d for d in device_lib.list_local_devices() if d.device_type == \'GPU\'];',
        'print(len(gpus));'
    )).read()
    return int(result)


class Process(object):

    def __init__(self, args, cluster, state=None):
        self._args = args
        self._cluster = cluster
        self._state = state
        listen_shutdown()


    def __call__(self, *args, **kwargs):
        try:
            self._initialize()

        except Exception as e:
            self._send_state(ready=False, error=e)
            logging.exception(e)


    def _initialize(self):
        args = self._args

        # worker device
        if args.backend == 'cpu':
            args.worker_device = "/job:worker/task:{}/{}:*".format(args.index, args.backend)
        else:
            # set gpu_id
            gpu_count = args.gpu_count if hasattr(args, 'gpu_count') else 1
            gpu_id = args.index % gpu_count
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            args.worker_device = "/job:worker/task:{}/{}:0".format(args.index, args.backend)

        # config
        config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            gpu_options=tf.GPUOptions(allow_growth=True)
        )

        server = tf.train.Server(self._cluster, job_name=args.job_name, task_index=args.index, config=config)
        if args.job_name == "ps":
            # send state
            self._send_state(ready=True)

            # wait
            while True: time.sleep(1000)

        else:
            worker = self._load_worker(server, args)
            # try n steps
            for i in range(3): worker()

            # send state
            self._send_state(ready=True)

            # do
            while True: worker()


    def _load_worker(self, server, args):
        assert self._args.worker_path is not None
        spec = importlib.util.spec_from_file_location("__importlib_module__", self._args.worker_path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        worker = m.create(server, args)
        return worker


    def _send_state(self, **kwargs):
        if self._state is None: return
        self._state.send(name=self._args.name, **kwargs)



class ProcessManager():
    def __init__(self):
        self.queue = multiprocessing.Queue()
        self.processes = {}

    def get_process(self, index): return self.processes[index]

    def create_process(self, args, cluster, wait=False):
        if args.name in self.processes: raise Exception('repeated process name {}'.format(args.name))

        def worker(args, cluster, pmgr):
            option.init(args.name)
            Process(args, cluster, pmgr)()

        # cache sys args
        argv = sys.argv
        sys.argv = [sys.executable] + option.args_to_argv(args)

        p = multiprocessing.Process(
            name=args.name,
            target=worker,
            args=(args, cluster, self)
        )
        p.daemon = True
        p.start()
        self.processes[args.name] = edict(process=p, state=None)

        # restore argv
        sys.argv = argv

        # update kill command
        self.update_kill_command(args.log_dir)

        if wait:
            data = self.processes[args.name]
            while data.state is None: self.update()
        return p

    def update(self, block=True):
        try:
            s = self.queue.get(block)
            self.processes[s.name].state = s
            if s.error is not None:
                self.clean()
                raise Exception(s.error)
            else:
                p = self.processes[s.name].process
                logging.info('process {} init finish'.format(s.name))
        except:
            return


    def send(self, name, ready, error=None):
        self.queue.put(edict(
            name = name,
            ready = ready,
            error = error
        ))


    def clean(self):
        for p in self.processes:
            p.terminate()


    def update_kill_command(self, log_dir):
        # create stop command
        with open(os.path.join(log_dir, 'kill.sh'), 'w') as f:
            pids = '{} '.format(os.getpid())
            for k, v in self.processes.items(): pids += '{} '.format(v.process.pid)
            f.write('#!/bin/sh \n' +
                    'kill -9 {}'.format(pids))


def init():

    # init option
    option.init('init', clean_log=True)

    # parse args
    args = option.args

    # recreate log dir
    if not os.path.exists(args.log_dir): os.mkdir(args.log_dir)

    # get gpu count
    args.gpu_count = get_gpu_count()


    # create processes
    manager = ProcessManager()

    # ps
    pargs = copy.deepcopy(args)
    pargs.name = pargs.job_name = 'ps'
    manager.create_process(pargs, option.cluster, wait=True)


    # workers
    for i in range(args.num_workers):
        pargs = copy.deepcopy(args)
        pargs.name = 'worker_{}'.format(i)
        pargs.index = i
        pargs.job_name = 'worker'
        manager.create_process(pargs, option.cluster, wait=True)


    # finish
    logging.info('Finish all process init')
    while True: time.sleep(1.0)



if __name__ == '__main__':
    init()