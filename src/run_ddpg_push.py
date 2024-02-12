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
import os

# RLLab
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
os.environ["THEANO_FLAGS"] = "device=cpu"
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize

# RLLab policy import
from rllab.policies.gaussian_mlp_policy import DeterministicMLPPolicy
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.envs.normalized_env import NormalizedEnv
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction

# RLLab DDPG import
from rllab.algos.ddpg import DDPG
from rllab.misc.instrument import stub, run_experiment_lite
import itertools
from rllab import config

stub(globals())

from distutils.dir_util import copy_tree
import numpy as np
import os, shutil
# Assigning model directory
srcmodeldirs = ['../reinforcement/modelskip/']
modeldir = 'reinforcement/'
if os.path.exists(modeldir):
    shutil.rmtree(modeldir)

for srcdir in srcmodeldirs:
    copy_tree(srcdir, modeldir)


# Determining randomised colour of objects
def getcolor():

    color = np.random.uniform(low=0, high=1, size=3)
    while np.linalg.norm(color - np.array([1.,0.,0.])) < 0.5:
        color = np.random.uniform(low=0, high=1, size=3)
    return color


# Determining which object to push
def rand_push():
    # Looping
    while True:
        # Randomising object positions
        object_ = [np.random.uniform(low=-1.0, high=-0.4),
                     np.random.uniform(low=0.3, high=1.2)]
        goal = [np.random.uniform(low=-1.2, high=-0.8),
                     np.random.uniform(low=0.8, high=1.2)]
        if np.linalg.norm(np.array(object_)-np.array(goal)) > 0.45: break
    geoms = []
    for i in range(5):
        # Randomising start position
        pos_x = np.random.uniform(low=-0.9, high=0.9)
        pos_y = np.random.uniform(low=0, high=1.0)
        rgba = getcolor().tolist()
        isinv = 1 if np.random.random() > 0.5 else 0
        geoms.append([rgba+[isinv], pos_x, pos_y])
    vp = np.random.uniform(low=0, high=360)

    return dict(nvp=1, vp=vp, object=object_, goal=goal, imsize=(64, 64), geoms=geoms,
        name="push", modelname='Models/ctxskiprealtransform127_11751',
        meanfile=None, modeldata='Models/greenctxstartgoalvpdistractvalid.npy')


# Setting up 3DoF pusher environment
push_params = {
    "env" : "Pusher3DOF-v1",
    "rand" : rand_push,
}


# Defining modes
oracle_mode = dict(mode='oracle', mode2='oracle')
ours_mode = dict(mode='ours', mode2='ours', scale=1.0)
ours_recon = dict(mode='ours', mode2='oursrecon', scale=1.0, ablation_type='recon')
tpil_mode = dict(mode='tpil', mode2='tpil')
gail_mode = dict(mode='tpil', mode2='gail')
nofeat = dict(mode='ours', mode2='nofeat', scale=1.0, ablation_type='nofeat')
noimage = dict(mode='ours', mode2='noimage', scale=1.0, ablation_type='noimage')

# Setting seed for repeatability
seeds = [42]

sanity = None

for params in [push_params]:

    for nvar in range(10):
        randparams = params['rand']()

        for modeparams in [ours_mode]:

            for scale in [0.1, 1.0, 10.0]:
                # Updating parameters
                copyparams = randparams.copy()
                copyparams.update(modeparams)
                copyparams['scale'] = scale
                # Defining MDP
                mdp = normalize(GymEnv(params['env'], **copyparams))

                if copyparams['mode'] == 'tpil':
                    if sanity == 'change1':
                        copyparams = params['rand']()
                        copyparams.update(modeparams)
                        mdp2 = normalize(GymEnv(params['env'], **copyparams))
                    elif sanity == 'same':
                        mdp2 = mdp
                    elif sanity == 'changing':
                        mdp2 = normalize(GymEnv(params['env'], mode='tpil'))
                        
                if 'imsize' in copyparams:
                    imsize = copyparams['imsize']

                for seed in seeds:
                    # Gaussian MLP Policy
                    policy = DeterministicMLPPolicy(
                        env_spec=mdp.spec,
                        hidden_sizes=(64, 64),
                    )

                    baseline = LinearFeatureBaseline(
                        mdp.spec,
                    )

                    es = OUStrategy(env_spec=mdp.spec)

                    qf = ContinuousMLPQFunction(env_spec=mdp.spec)

                    batch_size = 50*250

                    # DDPG Algorithm
                    algo = DDPG(
                        env=mdp,
                        policy=policy,
                        es=es,
                        qf=qf,
                        baseline=baseline,
                        batch_size=batch_size,
                        whole_paths=True,
                        max_path_length=50,
                        n_itr=250,
                        epoch_length=1000,
                        min_pool_size=10000,
                        n_epochs=1000,
                        discount=0.99,
                        scale_reward=0.01,
                        qf_learning_rate=1e-3,
                        policy_learning_rate=1e-4,
                        **copyparams
                    )

                # Running experiment
                run_experiment_lite(
                    algo.train(),
                    exp_prefix="sim_push_DDPG",
                    n_parallel=1,
                    snapshot_mode="all",
                    seed=seed,
                    mode="ec2_mujoco",
                )
