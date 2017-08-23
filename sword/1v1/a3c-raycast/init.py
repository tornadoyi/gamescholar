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
        print('process {} received signal {}: exiting'.format(os.getpid(), signal))
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
        self._state.send(index=self._args.index, **kwargs)



class ProcessManager():
    def __init__(self):
        self.queue = multiprocessing.Queue()
        self.processes = {}

    def get_process(self, index): return self.processes[index]

    def create_process(self, name, args, wait=False):
        p = multiprocessing.Process(
            name=name,
            target=lambda *args, **kwargs: Process(*args, **kwargs)(),
            args=(args, option.cluster, self)
        )
        p.start()
        self.processes[args.index] = edict(process=p, state=None)

        if wait:
            data = self.processes[args.index]
            while data.state is None: self.update()
        return p



    def update(self, block=True):
        try:
            s = self.queue.get(block)
            self.processes[s.index].state = s
            if s.error is not None:
                self.clean()
                raise Exception(s.error)
            else:
                p = self.processes[s.index].process
                logging.info('process {} init finish'.format(p.name))
        except:
            return


    def send(self, index, ready, error=None):
        self.queue.put(edict(
            index = index,
            ready = ready,
            error = error
        ))


    def clean(self):
        for p in self.processes:
            p.terminate()


def init():
    # init option
    option.init()

    # clear start log
    if os.path.exists(option.ALL_LOG_DIR): os.remove(option.ALL_LOG_DIR)
    if os.path.exists(option.ERROR_LOG_DIR): os.remove(option.ERROR_LOG_DIR)

    # parse args
    args = option.args

    # recreate log dir
    if not os.path.exists(args.log_dir): os.mkdir(args.log_dir)

    # get gpu count
    args.gpu_count = get_gpu_count()


    # create processes
    manager = ProcessManager()

    # ps
    argv = copy.deepcopy(args)
    argv.job_name = 'ps'
    manager.create_process('ps', argv, wait=True)


    # workers
    for i in range(args.num_workers):
        argv = copy.deepcopy(args)
        name = 'worker_{}'.format(i)
        argv.index = i
        argv.job_name = 'worker'
        manager.create_process(name, argv, wait=True)



    # finish
    logging.info('finish init and exit')



if __name__ == '__main__':
    init()