"""
PyTorch code for SAC-NF. Copied and modified from PyTorch code for SAC-NF (Mazoure et al., 2019): https://arxiv.org/abs/1905.06893
"""
from utils1 import LinearSchedule
from risk import distortion_de
import os
from rlkit.torch import pytorch_util as ptu
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils.sac import soft_update, hard_update
from model import NormalizingFlowPolicy, QNetwork, DeterministicPolicy,IQuantileMlp
from utils import save_checkpoint, load_checkpoint

def quantile_regression_loss(input, target, tau, weight):
    """
    input: (N, T)
    target: (N, T)
    tau: (N, T)
    """
    input = input.unsqueeze(-1)
    target = target.detach().unsqueeze(-2)
    tau = tau.detach().unsqueeze(-1)
    weight = weight.detach().unsqueeze(-2)
    expanded_input, expanded_target = torch.broadcast_tensors(input, target)
    L = F.smooth_l1_loss(expanded_input, expanded_target, reduction="none")  # (N, T, T)
    sign = torch.sign(expanded_input - expanded_target) / 2. + 0.5
    rho = torch.abs(tau - sign) * L * weight
    return rho.sum(dim=-1).mean()
def get_params(policy,flow_family):
    if flow_family in ['iaf','dsf','ddsf']:
        gaussian = policy.parameters()
        nf = policy.n_flow.transforms.parameters()
        return gaussian, nf
    gaussian, nf = [],[]
    for key,value in policy.named_parameters():
        if "n_flow" in key:
            nf.append(value)
        else:
            gaussian.append(value)
    return gaussian, nf


class SAC(object):
    """
    SAC class from Haarnoja et al. (2018)
    We leave the option to use automatice_entropy_tuning to avoid selecting entropy rate alpha
    """
    def __init__(self, num_inputs, action_space,args,
            risk_type='neutral',
            risk_param=0.,
            risk_param_final=None,
            risk_schedule_timesteps=1,):
        self.n_flow = args.n_flows
        self.num_inputs = num_inputs
        self.flow_family = args.flow_family

        self.args=args
        self.zf_criterion = quantile_regression_loss
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.tau_type = 'iqn'
        self.risk_type=risk_type
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.num_quantiles=32
        self._n_train_steps_total = 0
        self.device = torch.device("cuda:4" )
        self.risk_schedule = LinearSchedule(risk_schedule_timesteps, risk_param,
                                            risk_param if risk_param_final is None else risk_param_final)
        self.zf1 = IQuantileMlp(
            input_size=num_inputs + action_space.shape[0],
            output_size=1,
            num_quantiles=self.num_quantiles,
            hidden_sizes=[256, 256],
        ).to(device=self.device)
        self.zf2 = IQuantileMlp(
            input_size=num_inputs + action_space.shape[0],
            output_size=1,
            num_quantiles=self.num_quantiles,
            hidden_sizes=[256, 256],
        ).to(device=self.device)
        self.target_zf1 = IQuantileMlp(
            input_size=num_inputs + action_space.shape[0],
            output_size=1,
            num_quantiles=self.num_quantiles,
            hidden_sizes=[256, 256],
        ).to(device=self.device)
        self.target_zf2 = IQuantileMlp(
            input_size=num_inputs + action_space.shape[0],
            output_size=1,
            num_quantiles=self.num_quantiles,
            hidden_sizes=[256, 256],
        ).to(device=self.device)

        self.zf1_optimizer = Adam(self.zf1.parameters(), lr=args.lr)
        self.zf2_optimizer = Adam(self.zf2.parameters(), lr=args.lr)
        hard_update(self.target_zf1, self.zf1)
        hard_update(self.target_zf2, self.zf2)

        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

        self.policy = NormalizingFlowPolicy(num_inputs, action_space.shape[0], args.hidden_size,args.n_flows,args.flow_family,args).to(self.device)
        gaussian_params, nf_params = get_params(self.policy,self.flow_family)

        self.policy_optim = Adam(gaussian_params, lr=args.lr)
        self.nf_optim = Adam(nf_params, lr=args.actor_lr,weight_decay=args.reg_nf)

    def select_action(self, state, eval=False):
        """
        Select action for a state
        (Train) Sample an action from NF{N(mu(s),Sigma(s))}
        (Eval) Pass mu(s) through NF{}
        """
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if not eval:
            self.policy.train()
            action, _, _, _, _ = self.policy.evaluate(state)
        else:
            self.policy.eval()
            action, _, _, _, _ = self.policy.evaluate(state,eval=True)

        action = action.detach().cpu().numpy()
        return action[0]
    def map_action(self, state):
        """
        Select action for a state
        (Train) Sample an action from NF{N(mu(s),Sigma(s))}
        (Eval) Pass mu(s) through NF{}
        """
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        state = state.repeat(3000,1)
        self.policy.eval()
        action, x_t, zs = self.policy.mapaction(state)

        #zs = zs.detach().cpu().numpy()

        return zs
    def get_tau(self, actions, fp=None):
        if self.tau_type == 'fix':
            presum_tau = ptu.zeros(len(actions), self.num_quantiles) + 1. / self.num_quantiles
        elif self.tau_type == 'iqn':  # add 0.1 to prevent tau getting too close
            presum_tau = ptu.rand(len(actions), self.num_quantiles) + 0.1
            presum_tau /= presum_tau.sum(dim=-1, keepdims=True)
        tau = torch.cumsum(presum_tau, dim=1)  # (N, T), note that they are tau1...tauN in the paper
        with torch.no_grad():
            tau_hat = ptu.zeros_like(tau)
            tau_hat[:, 0:1] = tau[:, 0:1] / 2.
            tau_hat[:, 1:] = (tau[:, 1:] + tau[:, :-1]) / 2.
        return tau, tau_hat, presum_tau
    def update_parameters(self, memory, batch_size, updates):
        """
        Update parameters of SAC-NF
        Exactly like SAC, but keep two separate Adam optimizers for the Gaussian policy AND the NF layers
        .backward() on them sequentially
        """
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        obs = torch.FloatTensor(state_batch).to(self.device)
        next_obs = torch.FloatTensor(next_state_batch).to(self.device)
        actions = torch.FloatTensor(action_batch).to(self.device)
        rewards = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        # for visualization
        #with torch.no_grad():
        #    sample_size = 500
        #    _action, _logprob, _preact, _, _ = self.policy.evaluate(state_batch, num_samples=sample_size)
        #    _action = _action.cpu().detach()
        #    _preact = _preact.cpu().detach()
        #    _logprob = _logprob.view(batch_size, sample_size, -1).cpu().detach()
        #    info = {
        #         'action': _action,
        #         'preact': _preact,
        #         'logprob': _logprob,
        #    }
        info = {}

        ''' update critic '''
        with torch.no_grad():
            new_next_actions, next_state_log_pi, _,_,_ = self.policy.evaluate(next_obs)
            next_tau, next_tau_hat, next_presum_tau = self.get_tau( new_next_actions)
            target_z1_values= self.target_zf1(next_obs, new_next_actions,next_tau_hat)
            target_z2_values = self.target_zf2(next_obs, new_next_actions,next_tau_hat)
            min_qf_next_target = torch.min(target_z1_values, target_z2_values) - self.alpha * next_state_log_pi
            z_target = rewards + mask_batch * self.gamma * (min_qf_next_target)
        tau, tau_hat, presum_tau = self.get_tau(actions)
        z1_pred = self.zf1(obs, actions, tau_hat)
        z2_pred = self.zf2(obs, actions, tau_hat)
          # Two Q-functions to mitigate positive bias in the policy improvement step
        zf1_loss = self.zf_criterion(z1_pred, z_target, tau_hat, next_presum_tau)
        zf2_loss = self.zf_criterion(z2_pred, z_target, tau_hat, next_presum_tau)

        new_actions, log_pi, _,_,_ = self.policy.evaluate(obs)

        # update
        self.zf1_optimizer.zero_grad()
        zf1_loss.backward()
        self.zf1_optimizer.step()
        self.zf2_optimizer.zero_grad()
        zf2_loss.backward()
        self.zf2_optimizer.step()
        risk_param = self.risk_schedule(self._n_train_steps_total)

        if self.risk_type == 'VaR':
            tau_ = ptu.ones_like(rewards) * risk_param
            q1_new_actions = self.zf1(obs, new_actions, tau_)
            q2_new_actions = self.zf2(obs, new_actions, tau_)
        else:
            with torch.no_grad():
                new_tau, new_tau_hat, new_presum_tau = self.get_tau(obs, new_actions )
            z1_new_actions = self.zf1(obs, new_actions, new_tau_hat)
            z2_new_actions = self.zf2(obs, new_actions, new_tau_hat)
            if self.risk_type in ['neutral', 'std']:
                q1_new_actions = torch.sum(new_presum_tau * z1_new_actions, dim=1, keepdims=True)
                q2_new_actions = torch.sum(new_presum_tau * z2_new_actions, dim=1, keepdims=True)
                if self.risk_type == 'std':
                    q1_std = new_presum_tau * (z1_new_actions - q1_new_actions).pow(2)
                    q2_std = new_presum_tau * (z2_new_actions - q2_new_actions).pow(2)
                    q1_new_actions -= risk_param * q1_std.sum(dim=1, keepdims=True).sqrt()
                    q2_new_actions -= risk_param * q2_std.sum(dim=1, keepdims=True).sqrt()
            else:
                with torch.no_grad():
                    risk_weights = distortion_de(new_tau_hat, self.risk_type, risk_param)
                q1_new_actions = torch.sum(risk_weights * new_presum_tau * z1_new_actions, dim=1, keepdims=True)
                q2_new_actions = torch.sum(risk_weights * new_presum_tau * z2_new_actions, dim=1, keepdims=True)
        q_new_actions = torch.min(q1_new_actions, q2_new_actions)
        policy_loss = (self.alpha * log_pi - q_new_actions).mean()
        nf_loss = ((self.alpha * log_pi) - q_new_actions).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.policy_optim.step()

        self.nf_optim.zero_grad()
        nf_loss.backward()
        self.nf_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs

        # update target value fuctions
        if updates % self.target_update_interval == 0:
            soft_update(self.target_zf1, self.zf1, self.tau)
            soft_update(self.target_zf2, self.zf2, self.tau)
        return zf1_loss.item(), zf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item(), info

    def save_model(self, info):
        """
        Save the weights of the network (actor and critic separately)
        """
        # policy
        save_checkpoint({
            **info,
            'state_dict': self.policy.state_dict(),
            'optimizer' : self.policy_optim.state_dict(),
            }, self.args, filename='policy-ckpt.pth.tar')
        save_checkpoint({
            **info,
            #'state_dict': self.policy.state_dict(),
            'optimizer' : self.nf_optim.state_dict(),
            }, self.args, filename='nf-ckpt.pth.tar')

        # critic
        save_checkpoint({
            **info,
            'state_dict': self.zf1.state_dict(),
            'optimizer' : self.zf1_optimizer.state_dict(),
            }, self.args, filename='zf1-ckpt.pth.tar')
        save_checkpoint({
            **info,
            'state_dict': self.target_zf1.state_dict(),
            #'optimizer' : self.critic_optim.state_dict(),
            }, self.args, filename='target_zf1-ckpt.pth.tar')
        save_checkpoint({
            **info,
            'state_dict': self.zf2.state_dict(),
            'optimizer' : self.zf2_optimizer.state_dict(),
            }, self.args, filename='zf2-ckpt.pth.tar')
        save_checkpoint({
            **info,
            'state_dict': self.target_zf1.state_dict(),
            #'optimizer' : self.critic_optim.state_dict(),
            }, self.args, filename='target_zf2-ckpt.pth.tar')

    def load_model(self, args):
        """
        Jointly or separately load actor and critic weights
        """
        # policy
        load_checkpoint(
            model=self.policy,
            optimizer=self.policy_optim,
            opt=args,
            device=self.device,
            filename='policy-ckpt.pth.tar',
            )
        load_checkpoint(
            #model=self.policy,
            optimizer=self.nf_optim,
            opt=args,
            device=self.device,
            filename='nf-ckpt.pth.tar',
            )

        # critic
        load_checkpoint(
            model=self.zf1,
            optimizer=self.zf1_optimizer,
            opt=args,
            device=self.device,
            filename='zf1-ckpt.pth.tar',
            )
        load_checkpoint(
            model=self.target_zf1,
            #optimizer=self.critic_optim,
            opt=args,
            device=self.device,
            filename='target_zf1-ckpt.pth.tar',
            )
        load_checkpoint(
            model=self.zf2,
            optimizer=self.zf2_optimizer,
            opt=args,
            device=self.device,
            filename='zf2-ckpt.pth.tar',
            )
        load_checkpoint(
            model=self.target_zf2,
            #optimizer=self.critic_optim,
            opt=args,
            device=self.device,
            filename='target_zf2-ckpt.pth.tar',
            )
