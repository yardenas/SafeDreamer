import functools
import os

import embodied
import numpy as np
from gymnasium.wrappers.compatibility import EnvCompatibility


class SafeAdaptationEnvCompatibility(EnvCompatibility):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        cost = info["cost"] if "cost" in info.keys() else 0.0
        return obs, reward, cost, False, False, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.env.seed(seed)
        return self.env.reset(options=options), {}


class SafeAdaptationGym(embodied.Env):
    def __init__(
        self,
        env,
        platform="gpu",
        repeat=1,
        obs_key="image",
        render=False,
        size=(64, 64),
        camera=-1,
        mode="train",
        camera_name="vision",
    ):
        # TODO: This env variable is meant for headless GPU machines but may fail
        # on CPU-only machines.
        if platform == "gpu" and "MUJOCO_GL" not in os.environ:
            os.environ["MUJOCO_GL"] = "egl"

        import safe_adaptation_gym

        robot, task = env.split("_", 1)

        env = safe_adaptation_gym.make(
            robot_name=robot,
            task_name=task,
            rgb_observation=True,
            render_lidar_and_collision=False,
        )

        self._dmenv = SafeAdaptationEnvCompatibility(env)
        from . import from_gymnasium

        self._env = from_gymnasium.FromGymnasium(self._dmenv, obs_key=obs_key)
        self._render = render if mode == "train" else True
        self._size = size
        self._camera = camera
        self._camera_name = camera_name
        self._repeat = repeat
        self._mode = mode

    @property
    def repeat(self):
        return self._repeat

    @functools.cached_property
    def obs_space(self):
        spaces = self._env.obs_space.copy()
        if self._render:
            spaces["image"] = embodied.Space(np.uint8, self._size + (3,))
            if self._camera_name == "vision_front_back":
                spaces["image2"] = embodied.Space(np.uint8, self._size + (3,))

        return spaces

    @functools.cached_property
    def act_space(self):
        return self._env.act_space

    def step(self, action):
        for key, space in self.act_space.items():
            if not space.discrete:
                assert np.isfinite(action[key]).all(), (key, action[key])

        action = action.copy()
        if action["reset"]:
            obs = self._reset()
        else:
            reward = 0.0
            cost = 0.0
            for i in range(self._repeat):
                obs = self._env.step(action)
                reward += obs["reward"]
                if "cost" in obs.keys():
                    cost += obs["cost"]
                if obs["is_last"] or obs["is_terminal"]:
                    break
            obs["reward"] = np.float32(reward)
            if "cost" in obs.keys():
                obs["cost"] = np.float32(cost)
        return obs

    def _reset(self):
        obs = self._env.step({"reset": True})
        return obs

    def render(self):
        return self._dmenv.render()
