import numpy as np
from gym.spaces import Discrete, Box
import matplotlib.pyplot as plt

class DifferentialGame():
    def __init__(self, n_agents, max_step=25, render=True,
                 action_low=-10, action_high=10):
        self.max_step = max_step
        self.n_agents = n_agents
        assert self.n_agents == 2
        
        self.action_range = [action_low, action_high]
        lows = np.array([np.array([action_low]) for _ in range(self.n_agents)], dtype=np.float32)
        highs = np.array([np.array([action_high]) for _ in range(self.n_agents)], dtype=np.float32)
        self._action_spaces = [Box(low=lows[i], high=highs[i]) for i in range(self.n_agents)]
        self._observation_spaces = [Discrete(1) for _ in range(self.n_agents)]
        self.payoff = {}

        h1 = 0.8
        h2 = 1.
        s1 = 3.
        s2 = 1.
        x1 = -5.
        x2 = 5.
        y1 = -5.
        y2 = 5.
        c = 10.
        def max_f(a1, a2):
            f1 = h1 * (-(np.square(a1 - x1) / s1) - (np.square(a2 - y1) / s1))
            f2 = h2 * (-(np.square(a1 - x2) / s2) - (np.square(a2 - y2) / s2)) + c
            return max(f1, f2)
          
        self.payoff[0] = lambda a1, a2: max_f(a1, a2)
        self.payoff[1] = lambda a1, a2: max_f(a1, a2)
        self.rewards = np.zeros((self.n_agents,))
        if render:
          self.prepare_render()
        
    def prepare_render(self):
      x = np.linspace(-10,10,100)
      y = np.linspace(-10,10,100)
      x_pts, y_pts = np.meshgrid(x,y)
      
      vals = []
      for i in range(len(x)):
        _vals = []
        for j in range(len(y)):
          val = self.payoff[0](x_pts[i,j], y_pts[i,j])
          _vals.append(val)
        vals.append(_vals)
      states = np.array(vals)
      self.fig = plt.figure()
      
      ax = self.fig.add_subplot(111)
      cs = ax.contour(x, y, states, levels=[-30,-26,-22,-18,-14,-10,-6,-2])
      ax.clabel(cs, inline=True, fontsize=8)
      ax.set_xticks([-10,-5,0,5])
      ax.set_yticks([-10,-5,0,5])
      ax.set_xlabel('Action of Agent 1')
      ax.set_ylabel('Action of Agent 2')
      ax.grid('on')
      plt.ioff()
      
      self.lines = None

    def step(self, actions):
        assert len(actions) == self.n_agents
        self.t += 1
        
        actions = np.clip(actions, -1, 1)
        actions = np.array(actions).reshape((self.n_agents,)) * self.action_range[1]

        reward_n = np.zeros((self.n_agents,))
        for i in range(self.n_agents):
            reward_n[i] = self.payoff[i](*tuple(actions))
        self.rewards = reward_n
        
        state_n = np.array(list([[0] for i in range(self.n_agents)]))
        info = {}
        done_n = np.array([True] * self.n_agents)
        
        return state_n, reward_n, done_n, info

    def reset(self):
        self.t = 0
        return np.array(list([[0] for i in range(self.n_agents)]))
    
    def render(self, pts=None, close=False):
      if not plt.isinteractive():
        plt.ion()

      ax = self.fig.get_axes()[0]
      if len(ax.collections) > 8:
        ax.collections.pop()
        ax.lines = []
      ax.set_title("Reward %4.2f"%self.rewards[0])
      if (pts != None).all():
        ax.scatter(pts[:,0], pts[:,1])
        plt.pause(0.001)

    def get_joint_reward(self):
        return self.rewards
      
    @property
    def observation_space(self):
        return self._observation_spaces

    @property
    def action_space(self):
        return self._action_spaces
      
    @property
    def all_observation_space(self):
      return sum([obs_space.n for obs_space in self._observation_spaces])
