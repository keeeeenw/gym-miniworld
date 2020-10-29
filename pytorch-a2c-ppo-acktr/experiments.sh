#!/bin/bash

export DISPLAY=:0

# # Normal Learning rate meta
# python3 main.py --algo ppometa --num-frames 10000000 --num-processes 48 --num-steps 80 --lr 0.00005 --env-name MiniWorld-FourRoomsMeta-v0 --recurrent-policy

# # Normal Learning rate MiniWorld-FourRooms
# python3 main.py --algo ppometa --num-frames 10000000 --num-processes 48 --num-steps 80 --lr 0.00005 --env-name MiniWorld-FourRooms-v0 --recurrent-policy

# # Learning rate 0.00001
# python3 main.py --algo ppometa --num-frames 10000000 --num-processes 48 --num-steps 80 --lr 0.00001 --env-name MiniWorld-FourRoomsMeta-v0 --recurrent-policy

# # Learning rate 0.00003
# python3 main.py --algo ppometa --num-frames 10000000 --num-processes 48 --num-steps 80 --lr 0.00003 --env-name MiniWorld-FourRoomsMeta-v0 --recurrent-policy

# # Learning rate 0.0001
# python3 main.py --algo ppometa --num-frames 10000000 --num-processes 48 --num-steps 80 --lr 0.0001 --env-name MiniWorld-FourRoomsMeta-v0 --recurrent-policy

# # Learning rate 0.001
# python3 main.py --algo ppometa --num-frames 10000000 --num-processes 48 --num-steps 80 --lr 0.001 --env-name MiniWorld-FourRoomsMeta-v0 --recurrent-policy

# GRU 1
# python3 main.py --algo ppometa --num-frames 10000000 --num-processes 48 --num-steps 80 --lr 0.00005 --env-name MiniWorld-FourRoomsMeta-v0 --recurrent-policy --tb-dir one_gru

# GRU 1 MiniWorld-FourRooms
# python3 main.py --algo ppometa --num-frames 10000000 --num-processes 48 --num-steps 80 --lr 0.00005 --env-name MiniWorld-FourRooms-v0 --recurrent-policy --tb-dir one_gru_no_rand_color_room

# GRU 1 More Steps
# python3 main.py --algo ppometa --num-frames 50000000 --num-processes 48 --num-steps 80 --lr 0.00005 --env-name MiniWorld-FourRoomsMeta-v0 --recurrent-policy --tb-dir ppometa/one_gru_50M

# PPO MiniWorld-FourRooms
# python3 main.py --algo ppo --num-frames 10000000 --num-processes 48 --num-steps 80 --lr 0.00005 --env-name MiniWorld-FourRooms-v0 --recurrent-policy --tb-dir ppo_recur/one_gru_no_rand_color_room

# PPO Recur More Steps. Is this better due to more proccesses?
# python3 main.py --algo ppo --num-frames 50000000 --num-processes 48 --num-steps 80 --lr 0.00005 --env-name MiniWorld-FourRoomsMeta-v0 --recurrent-policy --tb-dir ppo_recur/one_gru_50M

# PPO MiniWorld-FourRooms-Meta 32 processes
#python3 main.py --algo ppo --num-frames 50000000 --num-processes 32 --num-steps 80 --lr 0.00005 --env-name MiniWorld-FourRoomsMeta-v0 --recurrent-policy --tb-dir ppo_recur/sanity_50M

# PPOMeta MiniWorld-FourRooms-Meta 32 processes
# python3 main.py --algo ppometa --num-frames 50000000 --num-processes 32 --num-steps 80 --lr 0.00005 --env-name MiniWorld-FourRoomsMeta-v0 --recurrent-policy --tb-dir ppo_meta/sanity_32_batch_50M
python3 main.py --algo ppometa --num-frames 50000000 --num-processes 48 --num-steps 80 --lr 0.00005 --env-name MiniWorld-FourRoomsMeta-v0 --recurrent-policy --tb-dir ppo_meta/sanity_48_step_50M
python3 main.py --algo ppometa --num-frames 50000000 --num-processes 32 --num-steps 80 --lr 0.00005 --env-name MiniWorld-FourRoomsMeta-v0 --recurrent-policy --tb-dir ppo_meta/sanity_32_step_50M

