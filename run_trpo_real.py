  
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

import os
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
# from rllab.envs.mujoco.gather.swimmer_gather_env import SwimmerGatherEnv
os.environ["THEANO_FLAGS"] = "device=cpu"
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize

from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.normalized_env import NormalizedEnv

from rllab.algos.trpo import DDPG
from rllab.misc.instrument import stub, run_experiment_lite
import itertools
from rllab import config

stub(globals())

from distutils.dir_util import copy_tree
import numpy as np
import os, shutil
srcmodeldirs = ['../ablation_data_paper/pushreal/']
modeldir = 'model/'
if os.path.exists(modeldir):
    shutil.rmtree(modeldir)

for srcdir in srcmodeldirs:
    copy_tree(srcdir, modeldir)


def rand_real():
    vp = np.random.uniform(low=0, high=360)
    vangle = np.random.uniform(low=-40, high=-70)
    cam_dist = np.random.uniform(low=1.5, high=2.5)
    distlow = 0.4
    distobj = np.random.uniform(low=distlow, high=0.7)
    distmult = np.random.uniform(low=1.7, high=2.1)
    object_ = [-(distobj - distlow), 0.0]
    goal = [-(distobj * distmult - distlow - 0.5), 0.0]
    return dict(vp=vp, vangle=vangle, object=object_, goal=goal,
        cam_dist=cam_dist, imsize=(48, 48), name="real",
        meanfile='Models/ctxskiprealtransform127_11751', modeldata='model/vdata_realnew200.npy')

real_params = {
    "env" : "Pusher3DOFReal-v1",
    "rand" : rand_real,
}

ours_mode = dict(mode='ours', mode2='ours', scale=0.01, modelname='model/pushreal_none/ablation_pushreal_None_30000')
ours_nofeat = dict(mode='ours', mode2='ours_nofeat', scale=0.01, ablation_type='nofeat', modelname='model/pushreal_none/ablation_pushreal_None_30000')
ours_noimage = dict(mode='ours', mode2='ours_noimage', scale=0.01, ablation_type='noimage', modelname='model/pushreal_none/ablation_pushreal_None_30000')
ab_l2 = dict(mode='ours', mode2='ab_l2', scale=0.01, modelname='model/pushreal_l2/ablation_pushreal_L2_30000')
ab_l2l3 = dict(mode='ours', mode2='ab_l2l3', scale=0.01, modelname='model/pushreal_l2l3/ablation_pushreal_L2L3_30000')
ab_l1 = dict(mode='ours', mode2='ab_l1', scale=0.01, modelname='model/pushreal_l1/ablation_pushreal_L1_30000')

seeds = [42]

for params in [real_params]:
    for nvar in range(10):
        randparams = params['rand']()
        for modeparams in [ab_l2]:#, ours_mode, ours_nofeat, ours_noimage, ab_l2l3, ab_l1]:
            copyparams = randparams.copy()
            copyparams.update(modeparams)
            mdp = normalize(GymEnv(params['env'], **copyparams))
            for seed in seeds:
                policy = GaussianMLPPolicy(
                    env_spec=mdp.spec,
                    hidden_sizes=(32, 32),
                    init_std=10
                )

                baseline = LinearFeatureBaseline(
                    mdp.spec,
                )

                batch_size = 50*250
                algo = DDPG(
                    env=mdp,
                    policy=policy,
                    baseline=baseline,
                    batch_size=batch_size,
                    whole_paths=True,
                    max_path_length=50,
                    n_itr=100,
                    step_size=0.01,
                    subsample_factor=1.0,
                    **copyparams
                )

                run_experiment_lite(
                    algo.train(),
                    exp_prefix="r-real-ab3",
                    n_parallel=4,
                    # dry=True,
                    snapshot_mode="all",
                    seed=seed,
                    mode="ec2_mujoco",
                    #terminate_machine=False
                )
