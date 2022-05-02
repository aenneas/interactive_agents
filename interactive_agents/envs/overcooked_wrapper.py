"""
Problems:
1) Overcooked agent index randomization?
2) Best hyperparams? How many episodes needed?
"""


from overcooked_ai_py.mdp.overcooked_env import Overcooked, OvercookedGridworld, OvercookedEnv
import numpy as np
import gym

class OvercookedWrapper:

    def __init__(self, env_config, spec_only=False):
        self._agent_id = env_config.get("agent_id", 0)
        self._other_agent_id = 1 - self._agent_id

        self.env_config = env_config.get("env_config")
        self.mdp_layout = env_config.get("layout_name")
        # self.mdp_rew_shaping_params = self.env_config.pop("rew_shaping_params")
        self.env_horizon = env_config.get("horizon")

        # Create and customize the Overcooked env.

        #First create OvercookedGridworld object from layout name.
        self.mdp = OvercookedGridworld.from_layout_name(self.mdp_layout)

        #OvercookedEnv from a given OvercookedGridworld.
        self.base_env = OvercookedEnv.from_mdp(self.mdp, horizon=self.env_horizon, info_level=0)

        #Use gym to create an Overcooked environment from their Gym wrapper.
        self.env = gym.make("Overcooked-v0")
        #Use custom_init to make the gym wrapper use the base env we created.
        #Lossless_state_encoding is the featurizing function for states.
        self.env.custom_init(self.base_env, self.base_env.lossless_state_encoding_mdp)


    @property
    def observation_space(self):
        return {self._agent_id: self.env.observation_space, self._other_agent_id: self.env.observation_space}

    @property
    def action_space(self):
        return {self._agent_id: self.env.action_space, self._other_agent_id: self.env.action_space}

    # IMPORTANT: Currently sample function in sampling.py goes through keys of obs dict to get all agent ids.
    # This is why we cannot have any other keys in obs dict. Either fix sample function or...
    def reset(self):
        _obs = self.env.reset()

        """Each agent's obs is of shape (Width, Height, Feature) where for each
        feature (Width,Height) is a binary map. We do rollaxis because CNN wants
        (Feature, Width, Height) 
        """
        _obs["both_agent_obs"] = (np.rollaxis(_obs["both_agent_obs"][0], 2),
                                  np.rollaxis(_obs["both_agent_obs"][1], 2))

        obs = {self._agent_id: _obs["both_agent_obs"][0], self._other_agent_id: _obs["both_agent_obs"][1]}
        return obs

    def step(self, action):
        _obs, reward, done, env_info = self.env.step(action)
        _obs["both_agent_obs"] = (np.rollaxis(_obs["both_agent_obs"][0],2),
                                  np.rollaxis(_obs["both_agent_obs"][1],2))
        obs = {self._agent_id: _obs["both_agent_obs"][0], self._other_agent_id: _obs["both_agent_obs"][1]}
        rewards = {self._agent_id: reward, self._other_agent_id: reward}
        dones = {self._agent_id: done, self._other_agent_id: done}
        return obs, rewards, dones, env_info

    def render(self, mode="human"):
        return self.render(mode)

"""
test_config = {"layout_name": "cramped_room", "horizon": 400}
test_overcooked = OvercookedWrapper(test_config)
test_obs = test_overcooked.reset()
print(test_obs[0].shape)
print(test_overcooked.observation_space)
print(test_overcooked.action_space)
"""