#!/usr/bin/env bash

# if num-workers=-1 means to eat up all the resources of machine
python start.py --session-name=a3c --num-workers=-1 --worker-path=a3c.py --backend=cpu