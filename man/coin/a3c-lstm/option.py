import argparse
from multiprocessing import cpu_count

def num_worker_type(v):
    v = int(v)
    return v if v > 0 else cpu_count()

parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument('--index', default=0, type=int, help='Task index')
parser.add_argument('--job-name', default="worker", help='worker or ps')
parser.add_argument('--num-workers', default=1, type=num_worker_type, help='Number of workers')
parser.add_argument('--backend', default='cpu', type=str, help='cpu, gpu')
parser.add_argument('--log-dir', type=str, default="./log", help="Log directory path")
parser.add_argument('--worker-path', default='', type=str, help='worker file path for load')
parser.add_argument('--session-name', type=str, default="project", help="session name")

args = parser.parse_args()

arg_dict = {}
for a in parser._actions:
    if not hasattr(args, a.dest): continue
    v = getattr(args, a.dest)
    arg_dict[a.option_strings[-1]] = v