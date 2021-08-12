#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: aithlab
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from differential_game import DifferentialGame
from replay_buffer import ReplayBuffer
from networks import Agent

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def random_actions(env, agents, replay_buffers):
  obs = env.reset()
  for _step in range(env.max_step):
    action_n = [agent.get_action(obs[i]) for i,agent in enumerate(agents)]
    action_n = np.asarray(action_n)
    
    next_observation_n, reward_n, done_n, info = env.step(action_n)
    replay_buffers.add(obs, action_n, reward_n, next_observation_n, done_n)
    obs = next_observation_n

buffer_size = 1e6
n_agents = 2
n_episodes = 200
batch_size = 512
max_step = 25
render = True
savedir = './figures'
os.makedirs(savedir, exist_ok=True)

env = DifferentialGame(n_agents, max_step=max_step, render=render)

replay_buffers = ReplayBuffer(buffer_size, n_agents, device)

obs_dims = [env.observation_space[i].n for i in range(n_agents)]
action_dims = [env.action_space[i].shape[0] for i in range(n_agents)]
agents = [Agent(obs_dims[i], action_dims[i], i, device=device) for i in range(n_agents)]


episode_actions, episode_mus, episode_opponent_mus, episode_rewards = [],[],[],[]
losses = {'Q':[], 'policy':[], 'opponent':[], 'prior':[]}
vals = {'log_prior':[], 'q_val':[], 'log_policy':[], 'log_opponent':[]}
while len(replay_buffers) <= batch_size:
  random_actions(env, agents, replay_buffers)

for episode in range(n_episodes):
  annealing = .1 + np.exp(-0.1 * max(episode, 0)) * 500.
  
  _losses = {'Q':[], 'policy':[], 'opponent':[], 'prior':[]}
  _vals = {'log_prior':[], 'q_val':[], 'log_policy':[], 'log_opponent':[]}
  rewards, _actions, _mus, _opponent_mus = [],[],[],[]
  obs = env.reset()
  for _step in range(env.max_step):
    mus = [agent.get_mu(obs[i]) for i, agent in enumerate(agents)]
    _mus.append(np.array(mus)[:,0])
    _opponent_mus.append(np.array(mus)[:,1])
    
    action_n = [agent.get_action(obs[i]) for i,agent in enumerate(agents)]
    action_n = np.asarray(action_n)
    _actions.append(action_n)
    next_observation_n, reward_n, done_n, info = env.step(action_n)
    
    replay_buffers.add(obs, action_n, reward_n, next_observation_n, done_n)
    obs = next_observation_n
    
    rewards.append(reward_n)
    
    agent_losses = {'Q':[], 'policy':[], 'opponent':[], 'prior':[]}
    agent_vals = {'log_prior':[], 'q_val':[], 'log_policy':[], 'log_opponent':[]}
      
    batch_n = replay_buffers.sample(batch_size)
      
    receent_indices = list(range(max(0, len(replay_buffers)-batch_size), len(replay_buffers)))
    recent_batch_n = replay_buffers.sample_by_indices(receent_indices)
    for i, agent in enumerate(agents):
      s, a, r, next_s, dones = batch_n[i]
      a_ = batch_n[1-i][1]
      
      recent_s, recent_oppo_a, _, _, _ = recent_batch_n[1-i]
      
      ## Prior update
      loss_prior, log_prior = agent.prior_update(recent_s, recent_oppo_a)  
      
      ## Opponent update
      loss_opponent, log_oppo = agent.opponent_model_update(s, annealing)
      
      ## Q update
      loss_q, q_val = agent.joint_q_update(s, a, a_, r)
      
      ## Policy Gradient
      loss_pg, log_policy = agent.policy_update(s, annealing)
      
      loss_policy = loss_opponent + loss_pg
      
      agent.optimizer_prior.zero_grad()
      loss_prior.backward()
      agent.optimizer_prior.step()

      agent.optimizer_policy.zero_grad()
      loss_policy.backward()
      agent.optimizer_policy.step()
      
      agent.optimizer_q.zero_grad()
      loss_q.backward()
      agent.optimizer_q.step()

      agent_losses['prior'].append(loss_prior.item())
      agent_losses['Q'].append(loss_q.item())
      agent_losses['policy'].append(loss_pg.item())
      agent_losses['opponent'].append(loss_opponent.item())
      
      agent_vals['log_prior'].append(log_prior.item())
      agent_vals['q_val'].append(q_val.item())
      agent_vals['log_policy'].append(log_policy.item())
      agent_vals['log_opponent'].append(log_oppo.item())

    _losses['Q'].append(agent_losses['Q'])
    _losses['policy'].append(agent_losses['policy'])
    _losses['opponent'].append(agent_losses['opponent'])
    _losses['prior'].append(agent_losses['prior'])

    _vals['log_prior'].append(agent_vals['log_prior'])
    _vals['q_val'].append(agent_vals['q_val'])
    _vals['log_policy'].append(agent_vals['log_policy'])
    _vals['log_opponent'].append(agent_vals['log_opponent'])
    
  episode_actions.append(_actions)
  episode_mus.append(np.mean(_mus,0))
  episode_opponent_mus.append(np.mean(_opponent_mus,0))
  episode_rewards.append([np.mean(rewards,0)[0], np.std(rewards,0)[0]])
  
  if not episode % 1 and render: 
    env.render(np.array(episode_actions[-1])*10)
    
  losses['Q'].append(np.mean(_losses['Q'],0))
  losses['policy'].append(np.mean(_losses['policy'],0))
  losses['opponent'].append(np.mean(_losses['opponent'],0))
  losses['prior'].append(np.mean(_losses['prior'],0))
  
  vals['log_prior'].append(np.mean(_vals['log_prior'],0))
  vals['q_val'].append(np.mean(_vals['q_val'],0))
  vals['log_policy'].append(np.mean(_vals['log_policy'],0))
  vals['log_opponent'].append(np.mean(_vals['log_opponent'],0))
  
  print("============="*5)
  print("| Ep. %d/%d | Reward %6.2f| Action %5.2f %5.2f | alpha %7.3f |" \
        %(episode, n_episodes, episode_rewards[-1][0], *episode_actions[-1][-1], annealing))
  
  for i in range(n_agents):
    print("| Agent #%d | Loss Q %6.3f PG %6.3f Opponent %6.3f Prior %6.3f |"\
          %(i+1, losses['Q'][-1][i], losses['policy'][-1][i], 
            losses['opponent'][-1][i], losses['prior'][-1][i]))
    print("| log P %6.3f | Q %6.3f | log pi %6.3f | log rho %6.3f |"\
          %(vals['log_prior'][-1][i], vals['q_val'][-1][i], 
            vals['log_policy'][-1][i], vals['log_opponent'][-1][i]))

losses['Q'] = np.array(losses['Q'])
losses['policy'] = np.array(losses['policy'])
losses['opponent'] = np.array(losses['opponent'])
losses['prior'] = np.array(losses['prior'])

episode_mus = np.array(episode_mus)*10
episode_opponent_mus = np.array(episode_opponent_mus)*10
episode_rewards = np.array(episode_rewards)

# Learning curve
fig = plt.figure(figsize=(14,7))
for i, key in enumerate(losses.keys()):
  ax = fig.add_subplot(1,4,i+1)
  ax.set_title(key)
  for j in range(n_agents):
    ax.plot(losses[key][:,j], label='Agent %d'%(j+1))
  plt.legend()
plt.savefig(os.path.join(savedir, 'learning_curve.png'))

# Return
episode_mean = episode_rewards[:,0]
episode_std = episode_rewards[:,1]
plt.figure(figsize=(8,5))
plt.plot(episode_mean)
plt.fill_between(np.arange(len(episode_mean)), 
                 episode_mean-episode_std, 
                 episode_mean+episode_std, alpha=0.5)
plt.ylabel('Return')
plt.xlabel('Episodes')
plt.xticks(np.arange(0,200+1,25))
plt.xlim(0,200)
plt.grid('on')
plt.savefig(os.path.join(savedir, 'Rewards.png'))

# Policies
plt.figure()
plt.plot(episode_mus[:,0], 'b', linestyle='dotted', label=r'$\mu_{\pi^1}$')
plt.plot(episode_mus[:,1], 'm', linestyle='dotted', label=r'$\mu_{\pi^2}$')
plt.plot(episode_opponent_mus[:,0], 'g', label=r'$\mu_{\rho^1}$')
plt.plot(episode_opponent_mus[:,1], 'r', label=r'$\mu_{\rho^2}$')
plt.xlim(0,200)
plt.legend(loc='lower right')
plt.ylabel('Mean of Policy')
plt.xlabel('Episodes')
plt.grid('on')
plt.savefig(os.path.join(savedir, 'policy.png'))