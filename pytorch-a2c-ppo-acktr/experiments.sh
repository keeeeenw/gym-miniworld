#!/bin/bash

export DISPLAY=:0

# Normal Learning rate meta
python3 main.py --algo ppometa --num-frames 10000000 --num-processes 48 --num-steps 80 --lr 0.00005 --env-name MiniWorld-FourRoomsMeta-v0 --recurrent-policy

# Normal Learning rate
python3 main.py --algo ppometa --num-frames 10000000 --num-processes 48 --num-steps 80 --lr 0.00005 --env-name MiniWorld-FourRooms-v0 --recurrent-policy

# Learning rate 0.00001
python3 main.py --algo ppometa --num-frames 10000000 --num-processes 48 --num-steps 80 --lr 0.00001 --env-name MiniWorld-FourRoomsMeta-v0 --recurrent-policy

# Learning rate 0.00003
python3 main.py --algo ppometa --num-frames 10000000 --num-processes 48 --num-steps 80 --lr 0.00003 --env-name MiniWorld-FourRoomsMeta-v0 --recurrent-policy

# Learning rate 0.0001
python3 main.py --algo ppometa --num-frames 10000000 --num-processes 48 --num-steps 80 --lr 0.0001 --env-name MiniWorld-FourRoomsMeta-v0 --recurrent-policy

# Learning rate 0.001
python3 main.py --algo ppometa --num-frames 10000000 --num-processes 48 --num-steps 80 --lr 0.001 --env-name MiniWorld-FourRoomsMeta-v0 --recurrent-policy