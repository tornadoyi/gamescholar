#!/bin/bash

name='a3c'

# stop
sh stop.sh

# create session
tmux new -s $name -n empty -d bash

# start
tmux new-window -t $name:1 -n $name -d bash
# if num-workers=-1 means to eat up all the resources of machine
tmux send-keys -t $name:$name "python init.py --num-workers=-1 --worker-path=$name.py --backend=gpu" Enter


# tensorboard
tmux new-window -t $name:2 -n tb -d bash
tmux send-keys -t $name:tb "tensorboard --logdir ./log" Enter

# htop
tmux new-window -t $name:3 -n htop -d bash
tmux send-keys -t $name:htop "htop" Enter
