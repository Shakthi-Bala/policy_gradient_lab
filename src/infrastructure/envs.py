"""Compatibility helpers for the assignment's pinned legacy environments."""

import gym


class LunarParticleCleanup(gym.Wrapper):
    """Remove visual-only exhaust after stepping, without changing task dynamics.

    Gym 0.25.2 ages these particles only in render(), so long headless episodes
    accumulate hundreds of unnecessary Box2D bodies. Their collision mask only
    includes static ground, not the lander or legs. Creation, RNG draws, engine
    impulses, observations and rewards are unchanged; cleanup happens afterward.
    """

    def step(self, action):
        result = self.env.step(action)
        self.env.unwrapped._clean_particles(True)
        return result


def make_env(name, prune_lunar_particles=False, **kwargs):
    env = gym.make(name, **kwargs)
    if prune_lunar_particles:
        if name != "LunarLander-v2":
            env.close()
            raise ValueError(
                "Particle cleanup is specific to LunarLander-v2 / Gym 0.25.2"
            )
        env = LunarParticleCleanup(env)
    return env
