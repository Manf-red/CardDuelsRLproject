import gymnasium as gym
from gymnasium import spaces


class ObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper that filters the observation space visible to the agent during training,
    but injects the complete state into the 'info' dictionary.
    """
    def __init__(self, env: gym.Env, obs_features: list[str] | None = None):
        super().__init__(env)
        self.obs_features = obs_features
        
        active_spaces = {
            k: v for k, v in env.observation_space.spaces.items() 
            if k in obs_features
        }
        self.observation_space = spaces.Dict(active_spaces)
        
    def observation(self, observation: dict) -> dict:
        """Apply the filtering mask to the raw state."""
        return {k: v for k, v in observation.items() if k in self.obs_features}
        
    def step(self, action: int):
        """
        Intercept the MDP transition.
        Save the unfiltered state in the info dictionary.
        """
        raw_obs, reward, terminated, truncated, info = self.env.step(action)
        
        info["full_state"] = raw_obs 
        
        return self.observation(raw_obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        raw_obs, info = self.env.reset(**kwargs)
        info["full_state"] = raw_obs
        
        return self.observation(raw_obs), info

    @property
    def board(self):
        """
        Guarantee transparency of the custom 'board' attribute by directly querying the underlying MDP environment.
        """
        return self.unwrapped.board
