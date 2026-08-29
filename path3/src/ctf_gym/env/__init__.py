from ctf_gym.env.base import BaseCTFEnv, CTFEnv
from ctf_gym.env.docker_env import DockerEnv
from ctf_gym.env.repl_env import ReplEnv

__all__ = ["CTFEnv", "BaseCTFEnv", "DockerEnv", "ReplEnv"]


def load_daytona_env():
    """Lazily import the Daytona backend so core works without daytona-sdk."""
    from ctf_gym.env.daytona_env import DaytonaEnv

    return DaytonaEnv


__all__.append("load_daytona_env")
