#!/bin/bash

host=120.92.33.192
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
