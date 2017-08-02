#!/usr/bin/expect -f
#spawn scp -r ./ guyi@192.168.8.103:/home/guyi/Projects/gamescholar
spawn rsync --progress -r  \
    --exclude .* --exclude *.pyc --exclude __pycache__ --exclude log \
    ./ guyi@192.168.8.103:/home/guyi/Projects/gamescholar

expect "*password:"
send "123456\r"

expect eof%