import os
import sys
import multiprocessing
import time
import copy
import logging
from easydict import EasyDict as edict
import option
import process


def get_gpu_count():
    result = os.popen('python -c \"{} {} {}\"'.format(
        'from tensorflow.python.client import device_lib;',
        'gpus=[d for d in device_lib.list_local_devices() if d.device_type == \'GPU\'];',
        'print(len(gpus));'
    )).read()
    return int(result)



class ProcessManager():
    def __init__(self):
        self.queue = multiprocessing.Queue()
        self.processes = {}


    def create_process(self, args, cluster, wait=False):
        if args.name in self.processes: raise Exception('repeated process name {}'.format(args.name))

        def worker(args, cluster, pmgr):
            option.init(args.name)
            process.Process(args, cluster, pmgr)()

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
        self.processes[args.name] = edict(process=p, args=args, state=None)

        # restore argv
        sys.argv = argv

        # update kill command
        self.update_kill_command(args.log_dir)

        if wait:
            data = self.processes[args.name]
            while data.state is None: self._update()
        return p


    def join(self, sleep=1.0):
        while True:
            alive_worker = False
            for _, p in self.processes.items():
                if p.args.job_name != 'worker': continue
                if not p.process.is_alive(): continue
                alive_worker = True
                break

            if not alive_worker:
                self.clean()
                break
            else:
                time.sleep(sleep)


    def send(self, name, state, error=None):
        self.queue.put(edict(
            name = name,
            state = state,
            error = error
        ))

    def _update(self, block=True):
        s = self.queue.get(block)
        p = self.processes[s.name]
        p.state = s.state
        p.error = s.error

        if s.error is not None:
            self.clean()
            raise Exception(s.error)
        else:
            logging.info('process {} {}'.format(s.name, p.state))


    def clean(self):
        for k, v in self.processes.items():
            v.process.terminate()


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
    logging.info('Finish all processes init')
    manager.join()
    logging.info('All processes end')


if __name__ == '__main__':
    init()