import tensorflow as tf
import numpy as np
import threading
from manager import Manager, init_global_bullet_policy



NUM_WORKERS = 8

def run(render=False):
    sess = tf.InteractiveSession()

    init_global_bullet_policy()

    managers = []
    for i in range(NUM_WORKERS):
        manager = Manager(sess, i)
        managers.append(manager)

    # init variables
    sess.run(tf.global_variables_initializer())


    worker_threads = []
    for i in range(len(managers)):
        manager = managers[i]
        if len(managers) > 1 and i == 0:
            job = lambda: manager.test(render=render)
        else:
            job = lambda: manager.train()

        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)


    # wait
    COORD = tf.train.Coordinator()
    COORD.join(worker_threads)



if __name__ == '__main__':
    run()