#!/bin/bash

host=192.168.8.103
username=guyi
password=123456
path=/home/guyi/Projects/


dstpath=$path`pwd | awk -F '/' '{print $NF}'`

expect -c "
    spawn rsync --progress -r --exclude .* --exclude *.pyc --exclude __pycache__ --exclude log ./ ${username}@${host}:${dstpath}

    expect \"*password:\"
    send \"${password}\r\"

    expect eof%
"
