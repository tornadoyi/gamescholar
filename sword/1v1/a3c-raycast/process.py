import os
import time
import platform
import logging
import importlib.util
import tensorflow as tf


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



class Process(object):

    def __init__(self, args, cluster, manager=None):
        self._args = args
        self._cluster = cluster
        self._manager = manager
        listen_shutdown()


    def __call__(self, *args, **kwargs):
        try:
            self._initialize()

        except Exception as e:
            self._notify(state='running', error=e)
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
            self._notify(state='running')

            # wait
            while True: time.sleep(1000)

        else:
            generator = self._load_worker(server, args)

            try:
                # try n steps
                for i in range(3): next(generator)

                # send state
                self._notify(state='running')

                # do
                while True: next(generator)

            except StopIteration:
                self._notify(state='end')


    def _load_worker(self, server, args):
        assert self._args.worker_path is not None
        spec = importlib.util.spec_from_file_location("__importlib_module__", self._args.worker_path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        g = m.main(server, args)
        return g


    def _notify(self, **kwargs):
        if self._manager is None: return
        self._manager.send(name=self._args.name, **kwargs)




if __name__ == '__main__':
    import option
    option.init()
    Process(option.args, option.cluster)()
