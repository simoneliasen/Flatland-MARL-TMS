import numpy as np
from collections import defaultdict
from typing import Dict, Any, Optional, Set, List, Tuple
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.step_utils.states import TrainState
from flatland.envs.rail_env import RailEnv, RailEnvActions

#Inspired from:
#https://gitlab.aicrowd.com/flatland/flatland/-/blob/master/flatland/contrib/wrappers/flatland_wrappers.py


class RailEnvWrapper:
  def __init__(self, env:RailEnv):
    self.env = env

    assert self.env is not None
    assert self.env.rail is not None, "Reset original environment first!"
    assert self.env.agents is not None, "Reset original environment first!"
    assert len(self.env.agents) > 0, "Reset original environment first!"

  # @property
  # def number_of_agents(self):
  #   return self.env.number_of_agents
  
  # @property
  # def agents(self):
  #   return self.env.agents

  # @property
  # def _seed(self):
  #   return self.env._seed

  # @property
  # def obs_builder(self):
  #   return self.env.obs_builder

  def __getattr__(self, name):
    try:
      return super().__getattr__(self,name)
    except:
      """Expose any other attributes of the underlying environment."""
      return getattr(self.env, name)


  @property
  def rail(self):
    return self.env.rail
  
  @property
  def width(self):
    return self.env.width
  
  @property
  def height(self):
    return self.env.height

  @property
  def agent_positions(self):
    return self.env.agent_positions

  def get_num_agents(self):
    return self.env.get_num_agents()

  def get_agent_handles(self):
    return self.env.get_agent_handles()

  def step(self, action_dict: Dict[int, RailEnvActions]):
    return self.env.step(action_dict)

  def reset(self, **kwargs):
    obs, info = self.env.reset(**kwargs)
    return obs, info




def find_all_cells_where_agent_can_choose(env: RailEnv):
    """
    input: a RailEnv (or something which behaves similarly, e.g. a wrapped RailEnv),
    WHICH HAS BEEN RESET ALREADY!
    (o.w., we call env.rail, which is None before reset(), and crash.)
    """
    switches = []
    switches_neighbors = []
    directions = list(range(4))
    for h in range(env.height):
        for w in range(env.width):
            pos = (h, w)

            is_switch = False
            # Check for switch: if there is more than one outgoing transition
            for orientation in directions:
                possible_transitions = env.rail.get_transitions(*pos, orientation)
                num_transitions = np.count_nonzero(possible_transitions)
                if num_transitions > 1:
                    switches.append(pos)
                    is_switch = True
                    break
            if is_switch:
                # Add all neighbouring rails, if pos is a switch
                for orientation in directions:
                    possible_transitions = env.rail.get_transitions(*pos, orientation)
                    for movement in directions:
                        if possible_transitions[movement]:
                            switches_neighbors.append(get_new_position(pos, movement))

    decision_cells = switches + switches_neighbors
    return tuple(map(set, (switches, switches_neighbors, decision_cells)))



class SkipNoChoiceCellsWrapper(RailEnvWrapper):
  
    # env can be a real RailEnv, or anything that shares the same interface
    # e.g. obs, rewards, dones, info = env.step(action_dict) and obs, info = env.reset(), and so on.
    def __init__(self, env:RailEnv, accumulate_skipped_rewards: bool, discounting: float) -> None:
      super().__init__(env)
      # save these so they can be inspected easier.
      self.accumulate_skipped_rewards = accumulate_skipped_rewards
      self.discounting = discounting
      self.switches = None
      self.switches_neighbors = None
      self.decision_cells = None
      self.skipped_rewards = defaultdict(list)

      # sets initial values for switches, decision_cells, etc.
      self.reset_cells()


    def on_decision_cell(self, agent: EnvAgent) -> bool:
      return agent.position is None or agent.position == agent.initial_position or agent.position in self.decision_cells

    def on_switch(self, agent: EnvAgent) -> bool:
      return agent.position in self.switches

    def next_to_switch(self, agent: EnvAgent) -> bool:
      return agent.position in self.switches_neighbors

    def reset_cells(self) -> None:
      self.switches, self.switches_neighbors, self.decision_cells = find_all_cells_where_agent_can_choose(self.env)

    def step(self, action_dict: Dict[int, RailEnvActions]) -> Tuple[Dict, Dict, Dict, Dict]:
      o, r, d, i = {}, {}, {}, {}
    
      # need to initialize i["..."]
      # as we will access i["..."][agent_id]
      i["action_required"] = dict()
      i["malfunction"] = dict()
      i["speed"] = dict()
      i["state"] = dict()

      while len(o) == 0:
        obs, reward, done, info = self.env.step(action_dict)

        for agent_id, agent_obs in obs.items():
          if done[agent_id] or self.on_decision_cell(self.env.agents[agent_id]):
            o[agent_id] = agent_obs
            r[agent_id] = reward[agent_id]
            d[agent_id] = done[agent_id]

            i["action_required"][agent_id] = info["action_required"][agent_id] 
            i["malfunction"][agent_id] = info["malfunction"][agent_id]
            i["speed"][agent_id] = info["speed"][agent_id]
            i["state"][agent_id] = info["state"][agent_id]
                                                          
            if self.accumulate_skipped_rewards:
              discounted_skipped_reward = r[agent_id]

              for skipped_reward in reversed(self.skipped_rewards[agent_id]):
                discounted_skipped_reward = self.discounting * discounted_skipped_reward + skipped_reward

              r[agent_id] = discounted_skipped_reward
              self.skipped_rewards[agent_id] = []

          elif self.accumulate_skipped_rewards:
            self.skipped_rewards[agent_id].append(reward[agent_id])
          # end of for-loop

        d['__all__'] = done['__all__']
        action_dict = {}
        # end of while-loop

      return o, r, d, i
        

    def reset(self, **kwargs) -> Tuple[Dict, Dict]:
      obs, info = self.env.reset(**kwargs)
      # resets decision cells, switches, etc. These can change with an env.reset(...)!
      # needs to be done after env.reset().
      self.reset_cells()
      return obs, info
