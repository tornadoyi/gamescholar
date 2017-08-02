import tensorflow as tf
import sys, signal
import time
import importlib.util
import logging



class Process(object):

    def __init__(self, args, cluster):
        self._args = args
        self._cluster = cluster

        self._listen_shutdown()



    def __call__(self):
        args = self._args

        # config
        config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            gpu_options=tf.GPUOptions(allow_growth=True)
        )

        server = tf.train.Server(self._cluster, job_name=args.job_name, task_index=args.index, config=config)
        if args.job_name == "worker":
            worker = self._load_worker()
            worker.run(server, args)

        else:
            while True:
                time.sleep(1000)


    def _listen_shutdown(self):
        def shutdown(signal, frame):
            print('Received signal %s: exiting' % signal)
            sys.exit(128+signal)

        signal.signal(signal.SIGHUP, shutdown)
        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)



    def _load_worker(self):
        assert self._args.worker_path is not None
        spec = importlib.util.spec_from_file_location("__importlib_module__", self._args.worker_path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m



def main(_):
    import option
    process = Process(option.args, option.cluster)
    try:
        process()

    except Exception as e:
        logging.ERROR(e)
        raise e



if __name__ == "__main__":
    tf.app.run()
