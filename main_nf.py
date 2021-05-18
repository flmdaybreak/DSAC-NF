"""
PyTorch code for SAC-NF. Copied and modified from PyTorch code for SAC-NF (Mazoure et al., 2019): https://arxiv.org/abs/1905.06893

To run:
    python main.py --env-name Ant-v2 --n_flows 3 --flow_family radial

For flows like IAF (tested) and DSF,DDSF (not tested), you need to install the `torchkit` package /repo (Huang et al., 2018)
"""
from rlkit.torch import pytorch_util as ptu
import os
cur_path = os.path.abspath(os.curdir)
import sys
import argparse
import time
import datetime
import itertools
import random
import pickle
import glob
 
import seaborn as sns
import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from sac_nf import SAC
from normalized_actions import NormalizedActions
from replay_memory import ReplayMemory
import pandas as pd
EvaluationReturn = 'remote_evaluation/Average Returns'
Qeval = 'q_critic_estimation'
Qmc = 'q_MC_real'
Qerr = 'q_error'
Epoch = 'Epoch'
test_num = 0
Qevaluations = []
try:
    import pybulletgym
    import sparse_gym_mujoco
except:
    print('No PyBullet Gym. Skipping...')
from utils.sac import get_params
from utils import logging, get_time, print_args
from utils import save_checkpoint, load_checkpoint

from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='PyTorch code for SAC-NF (Mazoure et al., 2019,https://arxiv.org/abs/1905.06893)')
parser.add_argument('--env-name', default="BipedalWalker-v3",
                    help='name of the environment to run')
parser.add_argument('--seed', type=int, default=2, metavar='N',
                    help='random seed (default: 456)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default:True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(tau) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--n_flows', type=int, default=2,
                    help='number of flows (default: 2)')
parser.add_argument('--actor_lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--flow_iterations', type=int, default=1,
                    help='number of NF iterations (default: 1)')
parser.add_argument('--flow_family', type=str, default='radial', metavar='G',
                    help='Flow family (planar,radial).')
parser.add_argument('--reg_nf', type=float, default=0, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--sigma', type=float, default=0, metavar='G',
                    help='sigma type (conditional=0,average=1,fixed=(0,+inf))')
parser.add_argument('--alpha', type=float, default=0.05, metavar='G',
                    help='Temperature parameter alpha determines the relative importance of the entropy term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Temperature parameter alpha automaically adjusted.')
#parser.add_argument('--seed', type=int, default=456, metavar='N',
#                    help='random seed (default: 456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--hadamard',type=int,default=1)
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--cache', default='experiments', type=str)
parser.add_argument('--experiment', default=None, help='name of experiment')
parser.add_argument('--nb_evals', type=int, default=10,
                    help='nb of evaluations')
parser.add_argument('--resume', dest='resume', action='store_true', default=True,
                    help='flag to resume the experiments')
parser.add_argument('--no-resume', dest='resume', action='store_false', default=True,
                    help='flag to resume the experiments')
parser.add_argument('--exp-num', type=int, default=0,
                    help='experiment number')

# seed


# log
parser.add_argument('--log-interval', type=int, default=1000,
                    help='log print-out interval (step)')
parser.add_argument('--eval-interval', type=int, default=10000,
                    help='eval interval (step)')
parser.add_argument('--map_interval', type=int, default=200000,
                    help='eval interval (step)')
parser.add_argument('--ckpt-interval', type=int, default=499999,
                    help='checkpoint interval (step)')
parser.add_argument('--gpu', type=int, default=4,
                    help='run on CUDA (default: False)')

args = parser.parse_args()
args.hadamard = bool(args.hadamard)
if args.gpu >= 0:
    print("gpu ok")
    ptu.set_gpu_mode(True, args.gpu)
# set env
if args.env_name == 'Humanoidrllab':
    from rllab.envs.mujoco.humanoid_env import HumanoidEnv
    from rllab.envs.normalized_env import normalize
    env = normalize(HumanoidEnv())
    max_episode_steps = float('inf')
    if args.seed >= 0:
        global seed_
        seed_ = args.seed
else:
    env = gym.make(args.env_name)
    max_episode_steps=env._max_episode_steps
    env=NormalizedActions(env)
    if args.seed >= 0:
        env.seed(args.seed)
if args.seed >= 0:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# set args
args.num_actions = env.action_space.shape[0]
args.max_action = env.action_space.high
args.min_action = env.action_space.low
args.num_flow_layers = args.n_flows
args.flowtype = args.flow_family

# set cache folder
if args.cache is None:
    args.cache = 'experiments'
if args.experiment is None:
    args.experiment = '-'.join(['sac-nf',
                                'mnf{}'.format(args.n_flows),
                                'sstep{}'.format(args.start_steps),
                                'a{}'.format(args.alpha),
                                'mlr{}'.format(args.lr),
                                'seed{}'.format(args.seed),
                                'exp{}'.format(args.exp_num),
                                ])
args.path = os.path.join(args.cache, args.experiment)
if args.resume:
    listing = glob.glob(args.path+'-19*') + glob.glob(args.path+'-20*')
    if len(listing) == 0:
        args.path = '{}-{}'.format(args.path, get_time())
    else:
        path_sorted = sorted(listing, key=lambda x: datetime.datetime.strptime(x, args.path+'-%y%m%d-%H:%M:%S'))
        args.path = path_sorted[-1]
        pass
else:
    args.path = '{}-{}'.format(args.path, get_time())
os.system('mkdir -p {}'.format(args.path))

# print args
logging(str(args), path=args.path)

# init tensorboard
writer = SummaryWriter(args.path)

# print config
configuration_setup='SAC-NF'
configuration_setup+='\n'
configuration_setup+=print_args(args)
#for arg in vars(args):
#    configuration_setup+=' {} : {}'.format(str(arg),str(getattr(args, arg)))
#    configuration_setup+='\n'
logging(configuration_setup, path=args.path)

# init sac
agent = SAC(env.observation_space.shape[0], env.action_space, args)
logging("----------------------------------------", path=args.path)
logging(str(agent.zf1), path=args.path)
logging("----------------------------------------", path=args.path)
logging(str(agent.policy), path=args.path)
logging("----------------------------------------", path=args.path)

gaussian_params, nf_params = get_params(agent.policy,args.flow_family)
nf_weights=sum(p.numel() for p in nf_params)
gaussian_weights = sum(p.numel() for p in gaussian_params)
logging('gaussian weights '+str(gaussian_weights), path=args.path)
logging('NF weights '+str(nf_weights), path=args.path)
logging('total weights '+str(nf_weights+gaussian_weights), path=args.path)


# memory
memory = ReplayMemory(args.replay_size)

# resume
args.start_episode = 1
args.offset_time = 0 # elapsed
args.total_numsteps = 0
args.updates = 0
args.eval_steps = 0
args.ckpt_steps = 0
agent.load_model(args)
memory.load(os.path.join(args.path, 'replay_memory'), 'pkl')

# Training Loop
total_numsteps = args.total_numsteps # 0
updates = args.updates # 0
map_steps = 0
eval_steps = args.eval_steps # 0
ckpt_steps = args.ckpt_steps # 0
start_episode = args.start_episode # 1
offset_time = args.offset_time # 0
start_time = time.time()
Qevaluations=[]
goal_path = cur_path + '/' +str(args.flow_family)+ '/' +str(args.env_name) + '/' + 'seed_' + str(args.seed)
os.makedirs(goal_path)
print(start_time,args.env_name)
if 'dataframe' in args:
    df = args.dataframe
else:
    df = pd.DataFrame(columns=["total_steps", "score_eval", "time_so_far"])

for i_episode in itertools.count(start_episode):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:
        if args.start_steps > total_numsteps:
            action = np.random.uniform(env.action_space.low,env.action_space.high,env.action_space.shape[0])  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy
        if len(memory) > args.start_steps:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                (critic_1_loss, critic_2_loss,
                 policy_loss,
                 _, _,
                 policy_info,
                 )= agent.update_parameters(memory, args.batch_size, updates)
                updates += 1

        else:
            pass

        next_state, reward, done, _ = env.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        eval_steps += 1
        ckpt_steps += 1
        map_steps +=1
        episode_reward += reward

        mask = 1 if episode_steps == max_episode_steps else float(not done)

        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state

    elapsed = round((time.time() - start_time + offset_time),2)
    logging("Episode: {}"
            ", time (sec): {}"
            ", total numsteps: {}"
            ", episode steps: {}"
            ", reward: {}"
            .format(
            i_episode,
            elapsed,
            total_numsteps,
            episode_steps,
            round(episode_reward, 2),
            ), path=args.path)
    #writer.add_scalar('train/ep_reward/episode', episode_reward, i_episode)
    #writer.add_scalar('train/ep_reward/step', episode_reward, total_numsteps)

    # evaluation
    if eval_steps>=args.eval_interval or total_numsteps > args.num_steps:
       # logging('evaluation time', path=args.path)
        r=[]
        for _ in range(args.nb_evals):
            state = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(state, eval=True)

                next_state, reward, done, _ = env.step(action)
                episode_reward += reward

                state = next_state
            r.append(episode_reward)
            Qevaluations.append([episode_reward, test_num])
            test_num += 1

        if (test_num +1 )>970:
            test = pd.DataFrame(columns=[EvaluationReturn, Epoch], data=Qevaluations)
            test.to_csv(goal_path + '/' + 'progress.csv')


        # writer
        writer.flush()

        # reset count
        eval_steps%=args.eval_interval
    if map_steps >args.map_interval:
        states = env.reset()
        actions = agent.map_action(states)
        ac0 = actions[0]
        ac0 = ac0.detach().cpu().numpy()
        ac00 = ac0[:,0]
        ac00 = ac00.astype(np.float64)
        ac05 = ac0[:, 5]
        ac05 = ac05.astype(np.float64)
        ac1 = actions[1]
        ac1 = ac1.detach().cpu().numpy()
        ac10 = ac1[:, 0]
        ac10 = ac10.astype(np.float64)
        ac15 = ac1[:, 5]
        ac15 = ac15.astype(np.float64)
        ac2 = actions[2]
        ac2 = ac2.detach().cpu().numpy()
        ac20 = ac2[:, 0]
        ac20 = ac20.astype(np.float64)
        ac25 = ac2[:, 5]
        ac25 = ac25.astype(np.float64)
        loc = goal_path + '/' + "numsteps:"+str(total_numsteps)+"-"
        ax = sns.kdeplot(ac00,
                         ac05,
                         cmap='Reds',
                         shade=True,
        shade_lowest = True )
        plt.xlabel("Action0[0]")
        plt.ylabel("Action0[5]")
        plt.savefig(loc+'action0.svg', format='svg', bbox_inches='tight')
        ax = sns.kdeplot(ac10,
                         ac15,
                         cmap='Reds',
                         shade=True,
        shade_lowest = True )
        plt.xlabel("Action1[0]")
        plt.ylabel("Action1[5]")
        plt.savefig(loc+'action1.svg', format='svg', bbox_inches='tight')
        ax = sns.kdeplot(ac20,
                         ac25,
                         cmap='Reds',
                         shade=True,
        shade_lowest = True )
        plt.xlabel("Action2[0]")
        plt.ylabel("Action2[5]")
        plt.savefig(loc+'action2.svg', format='svg', bbox_inches='tight')
        plt.clf()
        ax = sns.kdeplot(ac00, shade = True)
        plt.xlabel("Action0[0]")
        plt.savefig(loc+'action00d.svg', format='svg', bbox_inches='tight')

        plt.clf()
        ax = sns.kdeplot(ac05, shade = True)
        plt.xlabel("Action0[5]")
        plt.savefig(loc+'action05d.svg', format='svg', bbox_inches='tight')

        plt.clf()
        ax = sns.kdeplot(ac10, shade = True)
        plt.xlabel("Action1[0]")
        plt.savefig(loc+'action10d.svg', format='svg', bbox_inches='tight')

        plt.clf()
        ax = sns.kdeplot(ac15, shade = True)
        plt.xlabel("Action1[5]")
        plt.savefig(loc+'action15d.svg', format='svg', bbox_inches='tight')
        plt.clf()
        ax = sns.kdeplot(ac20, shade = True)
        plt.xlabel("Action2[0]")
        plt.savefig(loc+'action20d.svg', format='svg', bbox_inches='tight')
        plt.clf()
        ax = sns.kdeplot(ac25, shade = True)
        plt.xlabel("Action2[5]")
        plt.savefig(loc+'action25d.svg', format='svg', bbox_inches='tight')
        plt.clf()
        heatmap, xedges, yedges = np.histogram2d(ac20, ac25, bins=(80))
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.xlabel("Action2[0]")
        plt.ylabel("Action2[5]")
        plt.imshow(heatmap.T, extent=extent, origin='lower')
        plt.savefig(loc+'action2025d.svg', format='svg', bbox_inches='tight')
        plt.clf()
        map_steps %= args.map_interval

    if total_numsteps > args.num_steps:
        break

env.close()
