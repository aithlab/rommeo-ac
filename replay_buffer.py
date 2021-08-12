#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: aithlab
"""
import torch
import numpy as np

class ReplayBuffer():
  def __init__(self, size, n_agents, device=None):
    self._size = size
    self._n_agents = n_agents
    self._storages = [[] for _ in range(n_agents)]   
    self.device = device
    
  def __len__(self):
    assert len(self._storages[0]) == len(self._storages[1])
    return len(self._storages[0])
  
  def add(self, obs, action, reward, next_obs, done):
    for i in range(self._n_agents):
      self._storages[i].append((obs[i,0], action[i], reward[i], next_obs[i,0], done[i]))
      
      if len(self._storages[i]) > self._size:
        self._storages[i].pop(0)
    
  def get_samples(self, storage, idxs, _torch):
    obss, actions, rewards, next_obss, dones = [],[],[],[],[]
    for idx in idxs:
      obs, action, reward, next_obs, done = storage[idx]
      obss.append(obs)
      actions.append(action)
      rewards.append(reward)
      next_obss.append(next_obs)
      dones.append(done)
      
    if _torch:
      obss = torch.Tensor(np.array(obss)[:,None]).to(self.device)
      actions = torch.Tensor(np.array(actions)[:,None]).to(self.device)
      rewards = torch.Tensor(np.array(rewards)[:,None]).to(self.device)
      next_obss = torch.Tensor(np.array(next_obss)[:,None]).to(self.device)
      dones = torch.Tensor(np.array(dones)[:,None]).to(self.device)
      return obss, actions, rewards, next_obss, dones
    
    else:
      return np.array(obss)[:,None], np.array(actions)[:,None], \
             np.array(rewards)[:,None], np.array(next_obss)[:,None], np.array(dones)[:,None]
  
  def sample(self, batch_size, _torch=True, replace=True):
    buffers = []
    idxs = np.random.choice(self.__len__(), batch_size, replace=replace)
    for i in range(self._n_agents):
      obss, actions, rewards, next_obss, dones = self.get_samples(self._storages[i], idxs, _torch)
      buffers.append((obss, actions, rewards, next_obss, dones))
    return buffers
  
  def sample_by_indices(self, idxs, _torch=True, replace=True):
    buffers = []
    for i in range(self._n_agents):
      obss, actions, rewards, next_obss, dones = self.get_samples(self._storages[i], idxs, _torch)
      buffers.append((obss, actions, rewards, next_obss, dones))
    return buffers
  