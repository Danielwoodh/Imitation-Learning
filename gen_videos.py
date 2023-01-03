# #/  
# The MIT License (MIT)

# Copyright (c) 2016 rllab contributors

# rllab uses a shared copyright model: each contributor holds copyright over
# their contributions to rllab. The project versioning records all such
# contribution and copyright details.
# By contributing to the rllab repository through pull-request, comment,
# or otherwise, the contributor releases their content to the license and
# copyright terms herein.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Imports
from rllab.sampler.utils import rollout
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
import argparse
import os
import numpy as np
import joblib
import uuid
import pickle

filename = str(uuid.uuid4())


def all_videos(episode_number):
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('env', type=str,
                        help='environment')
    parser.add_argument('logdir', type=str,
                        help='logdir')
    parser.add_argument('--max_length', type=int, default=80,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=int, default=1,
                        help='Speedup')
    parser.add_argument('--loop', type=int, default=1,
                        help='# of loops')
    args = parser.parse_args()


    with open(args.file, 'rb') as pfile:
        policy = pickle.load(pfile)

    while True:

        env = normalize(GymEnv(args.env))
        env._wrapped_env.env.monitor.start(args.logdir, all_videos, force=False, resume=True)
        env._wrapped_env.env.monitor.configure(video_callable=all_videos)

        path = rollout(env, policy, max_path_length=args.max_length, animated=False, speedup=args.speedup)

        # Recording video for simulation
        vidpath = env._wrapped_env.env.monitor.video_recorder.path
        env._wrapped_env.env.monitor.close()

        if path is not None:
            true_rewards = np.sum(path['env_infos']['reward_true'])
            print(true_rewards)
