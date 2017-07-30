import tensorflow as tf
import sys, signal
import time
import importlib.util
import option


def _cluster_spec(num_workers, num_ps):
    """
More tensorflow setup for data parallelism
"""
    cluster = {}
    port = 12222

    all_ps = []
    host = '127.0.0.1'
    for _ in range(num_ps):
        all_ps.append('{}:{}'.format(host, port))
        port += 1
    cluster['ps'] = all_ps

    all_workers = []
    for _ in range(num_workers):
        all_workers.append('{}:{}'.format(host, port))
        port += 1
    cluster['worker'] = all_workers
    return cluster



class Process(object):

    def __init__(self):
        self._args = option.args
        self._cluster = _cluster_spec(self._args.num_workers, 1)

        self._listen_shutdown()



    def __call__(self):
        args = self._args

        server = tf.train.Server(self._cluster, job_name=args.job_name, task_index=args.index)
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
    process = Process()
    process()



if __name__ == "__main__":
    tf.app.run()
