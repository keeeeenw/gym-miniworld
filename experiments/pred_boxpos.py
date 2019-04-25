#!/usr/bin/env python3

import time
import random
import argparse
import math
import json
from functools import reduce
import operator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import gym
import gym_miniworld

from .utils import *

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.obs_to_out = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=6, stride=2),
            #nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Conv2d(32, 32, kernel_size=6, stride=2),
            #nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Conv2d(32, 16, kernel_size=6, stride=2),
            #nn.BatchNorm2d(16),
            nn.LeakyReLU(),

            #Print(),
            Flatten(),

            nn.Linear(384, 128),
            nn.LeakyReLU(),

            nn.Linear(128, 64),
            nn.LeakyReLU(),

            nn.Linear(64, 5),
            nn.Sigmoid(),
        )

        self.apply(init_weights)

        self.min = torch.cuda.FloatTensor((0.00, -0.00, -0.20, 0, 0))
        self.max = torch.cuda.FloatTensor((0.35, +0.15, +0.20, math.pi/2, 1))
        self.range = self.max - self.min

    def norm_to_full(self, v):
        return self.min + (v * self.range)

    def full_to_norm(self, v):
        return (v - self.min) / self.range

    def forward(self, obs):
        obs = obs / 255
        out = self.obs_to_out(obs)
        return self.norm_to_full(out)

def recon_test(env, model, gen_imgs=20):
    img_idx = 0
    while img_idx < gen_imgs:
        env.draw_static = True
        env.ergojr.draw_robot = True
        obs = env.reset()
        obs = obs.transpose(2, 1, 0)
        obs = make_var(obs).unsqueeze(0)

        pred_pos = model(obs)
        pred_pos = pred_pos.reshape(-1)
        pred_pos = pred_pos.detach().cpu().numpy()

        img_orig = env.render_obs()

        env.box.pos = pred_pos[0:3]
        env.box.dir = pred_pos[3]
        img_pred = env.render_obs()

        save_img('boxpos_{:03d}_orig.png'.format(img_idx), img_orig)
        save_img('boxpos_{:03d}_pred.png'.format(img_idx), img_pred)
        img_idx += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--buffer-size", default=200000, type=int)
    parser.add_argument("--weight-decay", default=0, type=float)
    parser.add_argument("--env", default="MiniWorld-BoxPos-v0")
    parser.add_argument("--model-path", default="pred_boxpos.torch")
    args = parser.parse_args()

    env = gym.make(args.env)

    num_actions = env.action_space.n
    print('num actions:', num_actions)

    max_steps = env.max_episode_steps
    print('max episode steps:', max_steps)

    # Done indicates that we become done after the current step
    buf_obs = np.zeros(shape=(args.buffer_size, 3, 80, 60), dtype=np.uint8)
    buf_pos = np.zeros(shape=(args.buffer_size, 5), dtype=np.float32)

    buf_num = 0
    cur_idx = 0

    model = Model()
    model.cuda()
    print_model_info(model)

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-4,
        weight_decay=args.weight_decay
    )

    def gen_data():
        global cur_idx, buf_num

        env.draw_static = True
        env.img_noise = True
        env.ergojr.draw_robot = True
        obs = env.reset()
        obs = obs.transpose(2, 1, 0)

        # Check that the box is visible
        env.sky_color = [0, 0, 0]
        env.draw_static = False
        env.img_noise = False
        env.ergojr.draw_robot = False
        seg = env.render_obs()
        box_present = np.any(seg)

        # Pick a random entry index. Prioritize expanding the set.
        #if buf_num < args.buffer_size and np.random.uniform(0, 1) < 0.5:
        if buf_num < args.buffer_size:
            cur_idx = buf_num
        else:
            cur_idx = np.random.randint(0, buf_num + 1) % args.buffer_size
        buf_num = max(buf_num, cur_idx+1)

        buf_obs[cur_idx] = obs
        buf_pos[cur_idx] = [*env.box.pos] + [env.box.dir] + [box_present]

    while buf_num <= args.batch_size:
        gen_data()

    running_loss = None

    start_time = time.time()

    for batch_no in range(1, 5000000):
        print('batch #{} (num entries={})'.format(batch_no, buf_num))

        batch_idx = np.random.randint(0, buf_num - args.batch_size)
        batch_obs = make_var(buf_obs[batch_idx:(batch_idx+args.batch_size)])
        batch_pos = make_var(buf_pos[batch_idx:(batch_idx+args.batch_size)])

        pred_pos = model(batch_obs)

        # Generate data while the GPU is computing
        for i in range(16):
            gen_data()

        # Box-present flag
        pred_bp = pred_pos[:, -1:]
        batch_bp = batch_pos[:, -1:]

        # Compute an L2 loss in the normalized range
        diff = model.full_to_norm(pred_pos) - model.full_to_norm(batch_pos)
        loss = (diff * diff)
        loss = torch.cat((loss[:,:4] * batch_bp, loss[:,-1:]), dim=1)
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_no == 1:
            running_loss = loss.data.detach().item()
        else:
            running_loss = 0.99 * running_loss + 0.01 * loss.data.detach().item()

        frame_count = batch_no * args.batch_size
        total_time = time.time() - start_time
        fps = int(frame_count / total_time)

        print('fps: {}'.format(fps))
        print('frames: {}'.format(frame_count))
        print('running loss: {:.5f}'.format(running_loss))

        acc_elems = (pred_bp > 0.5) == (batch_bp > 0.5)
        num_acc = acc_elems.sum().detach().item()
        acc = 100 * num_acc / pred_pos.size(0)
        print('box-present accuracy: {:.1f}%'.format(acc))

        if batch_no % 100 == 0:
            print('saving model')
            torch.save(model.state_dict(), args.model_path)

            recon_test(env, model)