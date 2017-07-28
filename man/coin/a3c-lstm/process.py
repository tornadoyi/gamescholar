import tensorflow as tf
import argparse
import sys, signal
import time
import os
import importlib.util

def _get_arg_parser():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--index', default=0, type=int, help='Task index')
    parser.add_argument('--job-name', default="worker", help='worker or ps')
    parser.add_argument('--num-workers', default=1, type=int, help='Number of workers')
    parser.add_argument('--backend', default='cpu', type=str, help='cpu, gpu')
    parser.add_argument('--worker-path', type=str, help='worker file path for load')
    return parser



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

    arg_parser = _get_arg_parser()

    def __init__(self):
        args = self.arg_parser.parse_args()

        self._index = args.index
        self._job_name = args.job_name
        self._num_workers = args.num_workers
        self._backend = args.backend
        self._worker_path = args.worker_path
        self._cluster = _cluster_spec(self._num_workers, 1)

        self._listen_shutdown()



    def __call__(self, *args, **kwargs):

        if self._job_name == "worker":
            server = tf.train.Server(self._cluster, job_name="worker", task_index=self._index)

            worker_device = "/job:worker/task:{}/{}:*".format(self._index, self._backend)
            config = tf.ConfigProto(device_filters=["/job:ps", worker_device])

            sess = tf.Session(server.target, config=config)
            worker = self._load_worker()
            worker.run(sess, self._index, worker_device)


        else:
            server = tf.train.Server(self._cluster, job_name="ps", task_index=self._index)
            while True:
                time.sleep(1000)




    def _listen_shutdown(self):
        def shutdown(signal, frame):
            print('Received signal %s: exiting', signal)
            sys.exit(128+signal)

        signal.signal(signal.SIGHUP, shutdown)
        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)



    def _load_worker(self):
        spec = importlib.util.spec_from_file_location("__importlib_module__", self._worker_path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m



def main(_):
    process = Process()
    process()



if __name__ == "__main__":
    tf.app.run()
