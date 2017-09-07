#!/bin/bash

name='a3c'

# stop
sh stop.sh

# create session
tmux new -s $name -n empty -d bash

# start  num-workers=-1 means to eat up all the resources of machine
tmux new-window -t $name:1 -n train -d bash
tmux send-keys -t $name:train "python init.py --num-workers=-1 --worker-path=$name.py --backend=cpu" Enter

# play for test
tmux new-window -t $name:2 -n play -d bash
tmux send-keys -t $name:play "python process.py --worker-path=$name.py --backend=cpu --index=98 --mode=play-online --auto-save=False" Enter

# tensorboard
tmux new-window -t $name:3 -n tb -d bash
tmux send-keys -t $name:tb "tensorboard --logdir ./log" Enter

# htop
tmux new-window -t $name:4 -n htop -d bash
tmux send-keys -t $name:htop "htop" Enter
