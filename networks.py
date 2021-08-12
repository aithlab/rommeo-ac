#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: aithlab

  1. policy 
  2. joint Q 
  3. opponent policy
  4. opponent prior 

"""
import torch
from torch import nn, optim
from torch.distributions import Normal, Independent

LOG_SIG_CAP_MAX = 2
LOG_SIG_CAP_MIN = -20

def squash_correction(actions):
  return (1 - torch.tanh(actions)**2 + 1e-6).log().sum(1)[:,None]

class Agent(nn.Module):
  def __init__(self, obs_dim, action_dim, agent_id,
               device='cpu', policy_lr=1e-2, q_lr=1e-2):
    super().__init__()
    self.device = device
    
    self.policy = PolicyNetwork(obs_dim, action_dim, agent_id).to(device) 
    self.initialize_weights(self.policy)
    
    self.joint_q = JointQNetwork(obs_dim, action_dim, agent_id).to(device) 
    self.initialize_weights(self.joint_q)
    
    self.opponent_prior = PriorNetwork(obs_dim, action_dim, agent_id).to(device) 
    self.initialize_weights(self.opponent_prior)

    self.opponent_model = OpponentNetwork(obs_dim, action_dim, agent_id).to(device) 
    self.initialize_weights(self.opponent_model)

    self.optimizer_q = optim.Adam(self.joint_q.parameters(), q_lr)
    params = [{'params':self.policy.parameters(), 'lr':policy_lr},
              {'params':self.opponent_model.parameters(), 'lr':policy_lr}]
    self.optimizer_policy = optim.Adam(params)
    self.optimizer_prior = optim.Adam(self.opponent_prior.parameters(), policy_lr)

  def get_action(self, s):
    s = torch.Tensor([s]).to(self.device)
    
    _actions, _, _,_ = self.opponent_model(s)
    
    action, _, _,_ = self.policy(s, _actions)

    return action.item() 
  
  def get_mu(self, s):
    s = torch.Tensor([s]).to(self.device)
    
    _, _, dist_,_ = self.opponent_model(s)
    mus_ = dist_.base_dist.loc
    
    _, _, dist,_ = self.policy(s, mus_)
    mu = dist.base_dist.loc

    return mu.item(), mus_.item()
  
  def initialize_weights(self, model):
    for n,p in model.named_parameters():
      if 'bias' in n:
        nn.init.constant_(p, 0)
      else:
        nn.init.xavier_normal_(p)
  
  def prior_update(self, s, a_):
    _, _, prior_dist, reg_loss = self.opponent_prior(s)
    
    raw_actions = torch.atanh(a_)
    log_pis = prior_dist.log_prob(raw_actions)[:,None]
    assert log_pis.shape == squash_correction(raw_actions).shape, \
    "%s %s"%(log_pis.shape, squash_correction(raw_actions).shape)
    log_pis = log_pis - squash_correction(raw_actions)
    
    loss = -log_pis.mean() + reg_loss
    
    return loss, log_pis.mean(), #prior_dist
  
  def joint_q_update(self, s, a, a_, r):
    q_values = self.joint_q(s, a, a_)
    assert q_values.shape == r.shape,  "%s %s"%(q_values.shape, r.shape)
    bellman_residual = 0.5*torch.mean((r - q_values)**2)
    
    return bellman_residual, q_values.mean()
  
  def opponent_model_update(self, s, annealing=1):
    opponent_actions, opponent_actions_log_pis, dist, reg_loss = self.opponent_model(s)
    raw_actions = torch.atanh(opponent_actions)
    
    _, _, prior_dist, _ = self.opponent_prior(s)
    prior_log_pis = prior_dist.log_prob(raw_actions)[:,None].detach()
    assert prior_log_pis.shape == squash_correction(raw_actions).shape, \
      "%s %s"%(prior_log_pis.shape, squash_correction(raw_actions).shape)
    prior_log_pis = prior_log_pis - squash_correction(raw_actions).detach()
        
    actions, agent_log_pis, _, _ = self.policy(s, opponent_actions)
  
    q_values = self.joint_q(s, actions, opponent_actions)
    assert opponent_actions_log_pis.shape == prior_log_pis.shape == \
           q_values.shape == agent_log_pis.shape, \
           "%s %s %s %s"%(opponent_actions_log_pis.shape, 
                    prior_log_pis.shape, q_values.shape, agent_log_pis.shape)
    
    loss = opponent_actions_log_pis.mean() - prior_log_pis.mean() \
          - q_values.mean() + annealing*agent_log_pis.mean() + reg_loss
    
    # check_list = [opponent_actions_log_pis.mean(), prior_log_pis.mean(), 
                  # q_values.mean(), agent_log_pis.mean(), reg_loss]
    return loss, opponent_actions_log_pis.mean()#, dist, check_list

  def policy_update(self, s, annealing=1):    
    opponent_actions, _, _, _ = self.opponent_model(s)
    
    actions, actions_log_pis, policy, reg_loss = self.policy(s, opponent_actions)
    
    q_values = self.joint_q(s, actions, opponent_actions)
    assert actions_log_pis.shape == q_values.shape, \
    "%s %s"%(actions_log_pis.shape, q_values.shape)
    
    loss = annealing*actions_log_pis.mean() - q_values.mean() + reg_loss
    
    # check_list = [actions_log_pis.mean(), q_values.mean(), reg_loss]
    return loss, actions_log_pis.mean()#, policy, check_list
  

class LinearSum(nn.Module):
  def __init__(self, in_dims, hidden_dim):
    super().__init__()
    self.build_network(in_dims, hidden_dim)

  def build_network(self, in_dims, hidden_dim):
    networks = nn.ModuleList()
    for in_dim in in_dims:
      networks.append(nn.Linear(in_dim, hidden_dim, bias=False))
    self.networks = networks
    self.bias = nn.Parameter(torch.zeros(hidden_dim), requires_grad=True)
    
  def forward(self, xs):
    outputs = 0
    for i, x in enumerate(xs):
      outputs = outputs + self.networks[i](x)
    assert outputs.shape[1:] == self.bias.shape, \
      "%s %s"%(outputs.shape, self.bias.shape)
    outputs = outputs + self.bias
    return outputs
          

class PriorNetwork(nn.Module):
  def __init__(self, obs_dim, action_dim, agent_id, 
               hidden_dims=[100,100], reg=1e-3):
    super().__init__()    
    in_dims = obs_dim
    self.build_network(in_dims, hidden_dims+[action_dim*2], agent_id)
    self.reg = reg
    
  def build_network(self, in_dim, hidden_dims, agent_id):
    self.network = nn.Sequential()
    for i, h_dim in enumerate(hidden_dims):
      self.network.add_module('layer%d_agent_%d'%(i, agent_id), nn.Linear(in_dim, h_dim))
      if i+1 != len(hidden_dims):
        self.network.add_module('activation_%d'%i, nn.ReLU())
      in_dim = h_dim
    
  def forward(self, s):
    x = self.network(s)
    mu, log_std = x.split([1,1],-1)
    log_std = log_std.clamp(LOG_SIG_CAP_MIN, LOG_SIG_CAP_MAX)
    
    reg_loss = self.reg*0.5*(log_std**2).mean() + self.reg*0.5*(mu**2).mean()
    
    # dist = Normal(mu, log_std.exp()+0.01)
    dist = Independent(Normal(mu, log_std.exp()+0.01), 1)
    
    actions = dist.rsample()
    log_prob = dist.log_prob(actions)[:,None]
    
    assert log_prob.shape == mu.shape == log_std.shape == actions.shape,\
      "%s %s %s %s"%(log_prob.shape, mu.shape, log_std.shape, actions.shape)
    
    return actions.tanh(), log_prob, dist, reg_loss
  

class OpponentNetwork(nn.Module):
  def __init__(self, obs_dim, action_dim, agent_id, 
               hidden_dims=[100,100], reg=1e-3):
    super().__init__()
    in_dims = obs_dim
    self.build_network(in_dims, hidden_dims+[action_dim*2], agent_id)
    self.reg = reg
    
  def build_network(self, in_dim, hidden_dims, agent_id):
    self.network = nn.Sequential()
    for i, h_dim in enumerate(hidden_dims):
      self.network.add_module('layer%d_agent_%d'%(i,agent_id), nn.Linear(in_dim, h_dim))
      if i+1 != len(hidden_dims):
        self.network.add_module('activation_%d'%i, nn.ReLU())
      in_dim = h_dim
    
  def forward(self, s):    
    x = self.network(s)
    mu, log_std = x.split([1,1],-1)
    log_std = log_std.clamp(LOG_SIG_CAP_MIN, LOG_SIG_CAP_MAX)
    
    reg_loss = self.reg*0.5*(log_std**2).mean() + self.reg*0.5*(mu**2).mean()
    
    # dist = Normal(mu, log_std.exp()+0.01)
    dist = Independent(Normal(mu, log_std.exp()+0.01), 1)
    
    actions = dist.rsample()
    log_prob = dist.log_prob(actions)[:,None] - squash_correction(actions)
    
    assert log_prob.shape == mu.shape == log_std.shape == actions.shape,\
      "%s %s %s %s"%(log_prob.shape, mu.shape, log_std.shape, actions.shape)
    
    return actions.tanh(), log_prob, dist, reg_loss

class JointQNetwork(nn.Module):
  def __init__(self, obs_dim, action_dim, agent_id, 
               hidden_dims=[100,100]):
    super().__init__()
    in_dims = [obs_dim, action_dim, action_dim]
    self.build_network(in_dims, hidden_dims+[1], agent_id)

  def build_network(self, in_dim, hidden_dims, agent_id):
    self.network = nn.Sequential()
    for i, h_dim in enumerate(hidden_dims):
      if i == 0:
        self.network.add_module('layer%d_agent_%d'%(i,agent_id), 
                                LinearSum(in_dim, h_dim))
      else:
        self.network.add_module('layer%d_agent_%d'%(i,agent_id), 
                                nn.Linear(in_dim, h_dim))
      if i+1 != len(hidden_dims):
        self.network.add_module('activation_%d'%i, 
                                nn.ReLU())
      in_dim = h_dim
  
  def forward(self, s, a, a_):
    x = self.network([s, a, a_])
    return x

class PolicyNetwork(nn.Module):
  def __init__(self, obs_dim, action_dim, agent_id, 
               hidden_dims=[100,100], reg=1e-3):
    super().__init__()
    in_dims = [obs_dim, action_dim]
    self.build_network(in_dims, hidden_dims+[action_dim*2], agent_id)
    self.reg = reg
    
  def build_network(self, in_dim, hidden_dims, agent_id):
    self.network = nn.Sequential()
    for i, h_dim in enumerate(hidden_dims):
      if i == 0:
        self.network.add_module('layer%d_agent_%d'%(i,agent_id), 
                                LinearSum(in_dim, h_dim))
      else:
        self.network.add_module('layer%d_agent_%d'%(i,agent_id), 
                                nn.Linear(in_dim, h_dim))
      if i+1 != len(hidden_dims):
        self.network.add_module('activation_%d'%i, 
                                nn.ReLU())
      in_dim = h_dim
    
  def forward(self, s, a_):
    x = self.network([s,a_])
    mu, log_std = x.split([1,1], -1)
    log_std = log_std.clamp(LOG_SIG_CAP_MIN, LOG_SIG_CAP_MAX)
    
    reg_loss = self.reg*0.5*(log_std**2).mean() + self.reg*0.5*(mu**2).mean()
    
    # dist = Normal(mu, log_std.exp()+0.01)
    dist = Independent(Normal(mu, log_std.exp()+0.01), 1)
    
    actions = dist.rsample()
    log_prob = dist.log_prob(actions)[:,None] - squash_correction(actions)
    
    assert log_prob.shape == mu.shape == log_std.shape == actions.shape,\
      "%s %s %s %s"%(log_prob.shape, mu.shape, log_std.shape, actions.shape)
    return actions.tanh(), log_prob, dist, reg_loss 
