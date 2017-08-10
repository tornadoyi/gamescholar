#!/usr/bin/expect -f

set host "120.92.33.192"
set username "guyi"
set password "123456"
set path "/home/guyi/Projects/gamescholar"


spawn rsync --progress -r --exclude .* --exclude *.pyc --exclude __pycache__ --exclude log ./ ${username}@${host}:${path}

expect "*password:"
send "${password}\r"

expect eof%