from dataclasses import dataclass, field
import random
import time
from typing import (
    Any,
)
from types import SimpleNamespace as Record

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

enu = enumerate

Tensor = torch.Tensor


@dataclass
class LinearSchedule:
    start: float
    end: float
    duration: int

    def __call__(self, t: int):
        slope = (self.end - self.start) / self.duration
        return self.end if t > self.duration else self.start + slope * t


@dataclass
class Step:
    obs: Any
    a: Any
    r: Any
    next_obs: Any
    term: bool
    trunc: bool
    info: Any


@dataclass
class GymEnv:
    id: str
    seed: int
    render_mode: str = None

    _env: Any = field(init=False)

    def __post_init__(self):
        env = gym.make(self.id, render_mode=self.render_mode)
        env.action_space.seed(self.seed) # XXX: When have to call this?
        self._env = env

    def actions(self):
        '''Set of possible actions'''
        return self._env.action_space

    def step(self, action):
        return self._env.step(action)

    def render(self):
        return self._env.render()

    def spawn(self):
        obs, _ = self._env.reset(seed=self.seed)
        return GymEpisode(obs=obs, env=self)


@dataclass
class GymEpisode:
    obs: Any
    env: GymEnv
    term: bool = False
    trunc: bool = False
    tot_rewards: float = 0.0

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)

        # Transition
        s = Step(
            obs=self.obs, # s you came from (s)
            a=action, # action (a)
            r=r, # reward from s, a -> (r)
            next_obs=obs, # s you went to (s')
            term=term, # terminated?
            trunc=trunc, # was truncated?
            info=info, # misc info
        )

        # Update this episode
        self.obs = obs
        self.term = term
        self.trunc = trunc
        self.tot_rewards += r

        return s

    @property
    def done(self):
        '''Somehow this episode is finished running'''
        return self.term or self.trunc


@dataclass
class GymCodec:
    env: Any

    obs_encoding_size: int = field(init=False)
    n_possible_actions: int = field(init=False)

    def __post_init__(self):
        env = self.env._env
        self.obs_encoding_size = np.array(env.observation_space.shape).prod()

        assert isinstance(env.action_space, gym.spaces.Discrete)
        self.n_possible_actions = env.action_space.n

    @property
    def encoded_obs_size(self):
        return self.obs_encoding_size

    @property
    def encoded_action_size(self):
        return self.n_possible_actions

    def encode_obs(self, obs) -> Tensor:
        return torch.Tensor(obs)


@dataclass
class ReplayBuffer:
    capacity: int # n of FIFO entries to support
    obs_size: int # Size of encoded observation representation

    # Internal attributes
    size: int = field(init=False) # How many entries actually exist?
    cursor: int = field(init=False) # Next index to write entry to
    obs: Tensor = field(init=False)
    action: Tensor = field(init=False)
    reward: Tensor = field(init=False)
    next_obs: Tensor = field(init=False)
    term: Tensor = field(init=False)

    def __post_init__(self):
        self.size = 0
        self.cursor = 0
        self.obs = torch.zeros((self.capacity, self.obs_size), dtype=torch.float)
        self.action = torch.zeros((self.capacity, 1), dtype=int)
        self.reward = torch.zeros((self.capacity, 1), dtype=torch.float)
        self.next_obs = torch.zeros((self.capacity, self.obs_size), dtype=torch.float)
        self.term = torch.zeros((self.capacity, 1), dtype=torch.float) # note how float

        assert str(self.obs.device) == "cpu", f"Actual device: '{self.obs.device}'"

    def add(self, step: Step):
        # XXX: why do rewards, actions need to be matrix and not vector?
        curs = self.cursor
        self.obs[curs] = torch.tensor(step.obs)
        self.action[curs] = step.a
        self.reward[curs] = step.r
        self.next_obs[curs] = torch.tensor(step.next_obs)
        self.term[curs] = step.term # bool will be converted to 0/1

        # Move cursor / update size
        self.size = min(self.size + 1, self.capacity)
        if self.cursor >= self.capacity - 1:
            self.cursor = 0
        else:
            self.cursor += 1

    def sample(self, n: int):
        # XXX: Don't create each batch each time. Overwrite pre-allocated.
        assert self.size > n
        ind = np.random.choice(a=self.size, size=n, replace=False)
        ind.sort() # :fingers-crossed: affects cache line?

        # Compose batch
        # - each attribute will be a tensor
        # - XXX: Need to send .to(device)?
        batch = Record()
        batch.obs = self.obs[ind]
        batch.next_obs = self.next_obs[ind]
        batch.reward = self.reward[ind]
        batch.action = self.action[ind]
        batch.term = self.term[ind]

        return batch


def sync_network(net1, net2):
    '''Copy net1's weights to net2'''
    for net2_param, net1_param in zip(net2.parameters(), net1.parameters()):
        net2_param.data.copy_(net1_param.data)


class QNetwork(nn.Module):

    def __init__(self, inp_size: int, out_size: int):
        super().__init__()
        # Input is size of a single observation
        # Output layer size is n actions poss in env
        # - Each output is a q_value.
        self.network = nn.Sequential(
            nn.Linear(inp_size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, out_size),
        )

    def forward(self, x):
        return self.network(x)


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_deterministic(determ: bool):
    torch.backends.cudnn.deterministic = determ


@dataclass
class DQNTrainer:
    env: Any
    codec: Any
    args: 'Args'

    def train(self):
        print("\nTraining")
        env = self.env
        codec = self.codec
        args = self.args

        # Handle determinism
        set_seeds(args.seed)
        set_deterministic(args.torch_deterministic)

        # Initialize Networks (Q and Target)
        device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
        inp_size, out_size = codec.encoded_obs_size, codec.encoded_action_size
        q_network = QNetwork(inp_size, out_size).to(device)
        target_network = QNetwork(inp_size, out_size).to(device)
        target_network.load_state_dict(q_network.state_dict())
        optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)

        # Train
        buffer = ReplayBuffer(capacity=args.buffer_size, obs_size=codec.encoded_obs_size)
        eps_sched = LinearSchedule(args.start_eps, args.end_eps, args.total_steps // 2)
        episode = env.spawn()
        r_ema, ema_a, max_r = 0.0, 0.05, 0.0
        for step_i in (pbar := tqdm(range(args.total_steps))):
            # Collect experiences
            # - Choose an action to take (either eps explore or on-policy)
            # - Take action -> advance env state
            # - Add transition to replay buffer
            # - If episode is over, reset it.
            if random.random() < eps_sched(step_i):
                action = env.actions().sample()
            else:
                obs_enc = codec.encode_obs(episode.obs).to(device)
                q_values = q_network(obs_enc)
                action = torch.argmax(q_values).item()
            step = episode.step(action)
            buffer.add(step)
            if episode.done:
                r_ema = ((1 - ema_a) * r_ema) + ema_a * episode.tot_rewards
                max_r = max(max_r, episode.tot_rewards)
                episode = env.spawn()

            # Train networks
            # - Update Q Network
            #   - Sample a batch of transitions from replay buffer.
            #   - Compute loss and update weights
            # - Sync Target Network
            #   - Copy the q network weights to target network after
            #     some specified timesteps.
            if step_i > args.train_start and step_i % args.train_freq == 0:
                samples = buffer.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(samples.next_obs).max(dim=1)
                    td_target = samples.reward.flatten() + args.gamma * target_max * (1 - samples.term.flatten())
                prev_val = q_network(samples.obs).gather(1, samples.action).squeeze()
                loss = F.mse_loss(td_target, prev_val)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Sync target network
                if step_i % args.target_update_freq == 0:
                    sync_network(q_network, target_network)

            # Report
            pbar.set_postfix({
                "ema(R)": r_ema,
                "max(R)": max_r,
            })

        return q_network

    def evaluate(
        self,
        q_network,
        n_episodes=10,
        eps=0.05,
        device=None,
    ):
        env = self.env
        codec = self.codec
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")

        episode = env.spawn()
        ep_rewards, max_r = 0.0, 0.0
        ep_i = 0
        while ep_i < n_episodes:
            if random.random() < eps:
                action = env.actions().sample()
            else:
                obs_enc = codec.encode_obs(episode.obs).to(device)
                q_values = q_network(obs_enc)
                action = torch.argmax(q_values).item()
            episode.step(action)
            if episode.done:
                ep_rewards += episode.tot_rewards
                max_r = max(max_r, episode.tot_rewards)
                episode = env.spawn()
                ep_i += 1
        just = 15
        print("\nEvaluation")
        print("max(R):".ljust(just) + f"{max_r:.1f}")
        print("E(R):".ljust(just) + f"{ep_rewards / n_episodes:.1f}")

    def render(self, q_network, eps=0.05, device=None):
        env = GymEnv(
            id=self.env.id,
            seed=self.env.seed,
            render_mode="rgb_array",
        )
        codec = self.codec
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")

        episode = env.spawn()
        frames = []
        frames.append(env.render())
        while True:
            if random.random() < eps:
                action = env.actions().sample()
            else:
                obs_enc = codec.encode_obs(episode.obs).to(device)
                q_values = q_network(obs_enc)
                action = torch.argmax(q_values).item()
            episode.step(action)
            frames.append(env.render())
            if episode.done:
                break

        Renderer.to_video(frames, env.id)


@dataclass
class Args:
    total_steps: int = 100_000
    train_start: int = 20_000
    train_freq: int = 10
    target_update_freq: int = 500
    batch_size: int = 64
    gamma: float = 0.99
    learning_rate: float = 2.5e-4
    start_eps: float = 1.0
    end_eps: float = 0.05
    buffer_size: int = 10_000
    gpu: bool = True
    seed: int = 1
    torch_deterministic: bool = True


class Renderer:

    @staticmethod
    def to_video(frames, name):
        import cv2
        height, width, layers = frames[0].shape
        size = (width, height)
        file_name = f'{name}.mp4'
        out = cv2.VideoWriter(
            file_name,
            cv2.VideoWriter_fourcc(*'mp4v'),
            30, size,
        )
        for frame in frames: # Convert RGB to BGR for OpenCV
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        print("\nRendered Video")
        print("- n frames:", len(frames))
        print("- Wrote file to:", file_name)


class Tasks:

    def inspect_gym_env(self):
        '''
        speed
        what do you have access to at each step?
        '''
        N = 20_000
        seed = 1
        env_id = "CartPole-v1"

        env = gym.make(env_id)
        env.action_space.seed(seed)
        st_time = time.time()
        obs, _ = env.reset(seed=seed)
        max_r = 0.0
        for _ in range(N):
            valid_actions = env.action_space
            action = valid_actions.sample()
            obs, r, term, trunc, info = env.step(action)
            max_r = max(r, max_r)
            if term or trunc:
                env.reset()
        print(max_r)
        elapsed = time.time() - st_time
        print("elapsed:", f"{elapsed:.2f}")
        print("rate:", f"{N / elapsed:.1f}")

    def check_render(self):
        env_id = "CartPole-v1"
        env = gym.make(env_id, render_mode="rgb_array")
        obs, _ = env.reset()

        frames = []
        frames.append(env.render())
        while True:
            valid_actions = env.action_space
            action = valid_actions.sample()
            obs, r, term, trunc, info = env.step(action)
            frames.append(env.render())
            if term or trunc:
                break
        Renderer.to_video(frames, "cartpole")

    def check_env(self):
        env_id = "CartPole-v1"
        seed = 1

        env = GymEnv(id=env_id, seed=seed)
        print("actions", env.actions)

        episode = env.spawn()
        while True:
            action = env.actions().sample()
            assert action in (0, 1)
            step = episode.step(action)
            print(step)
            print(episode)
            if episode.done:
                break

    def test_linear_sched(self):
        start, end, dur = 0.05, 1.0, 50
        sched = LinearSchedule(start, end, dur)
        assert sched(0) == start
        assert start <= sched(25) <= end
        assert sched(51) == end
        assert sched(200) == end
        for t in range(100):
            param = sched(t)
            print(t, param)

    def check_codec(self):
        env_id = "CartPole-v1"
        seed = 1

        env = GymEnv(env_id, seed=seed)
        codec = GymCodec(env)

        print(codec.encoded_obs_size)
        print(codec.encoded_action_size)

        episode = env.spawn()
        print(codec.encode_obs(episode.obs))

    def replay_buffer(self):
        env = GymEnv("CartPole-v1", seed=1)
        codec = GymCodec(env)
        buffer = ReplayBuffer(
            capacity=100,
            obs_size=codec.encoded_obs_size,
        )

        # Fill to over capacity
        episode = env.spawn()
        for _ in range(500):
            action = env.actions().sample()
            step = episode.step(action)
            buffer.add(step)
            if episode.done:
                episode = env.spawn()

        return buffer

    def check_replay_buffer(self):
        buffer = self.replay_buffer()

        # Sample a batch
        samples = buffer.sample(64)
        print(samples)

        return samples

    def check_q_network(self):
        # Single call
        # Vectorized call
        # Argmax results
        env = GymEnv("CartPole-v1", seed=1)
        codec = GymCodec(env)
        gamma = 0.99
        device = "cpu"
        inp_size, out_size = codec.encoded_obs_size, codec.encoded_action_size
        q_network = QNetwork(inp_size, out_size).to(device)

        # Policy call during sampling
        episode = env.spawn()
        obs_enc = codec.encode_obs(episode.obs).to(device)
        q_values = q_network(obs_enc)
        print(q_values)
        action = torch.argmax(q_values).item()
        print(action)

        # Training call
        buffer = self.replay_buffer()
        samples = buffer.sample(2)
        with torch.no_grad():
            q_vals = q_network(samples.next_obs)
            target_max, _ = q_vals.max(dim=1)
            td_target = samples.reward.flatten() + gamma * target_max * (1 - samples.term.flatten())
        prev_val = q_network(samples.obs).gather(1, samples.action).squeeze()
        print(q_vals)
        print(target_max)
        print(td_target)
        print(prev_val)


if __name__ == "__main__":
    # Tasks().inspect_gym_env()
    # Tasks().check_render()
    # Tasks().check_env()
    # Tasks().test_linear_sched()
    # Tasks().check_codec()
    # Tasks().check_replay_buffer()
    # Tasks().check_q_network()
    # import sys; sys.exit() # noqa

    ################
    # Train DQN
    ################
    env_id = "CartPole-v1"
    seed = 5
    env = GymEnv(env_id, seed=seed)
    codec = GymCodec(env)
    args = Args(
        total_steps=150_000,
        train_start=20_000,
        start_eps=1.00,
        end_eps=0.01,
        seed=seed,
    )
    trainer = DQNTrainer(env, codec, args)

    q_network = trainer.train()

    ################
    # Evaluate Model
    ################
    trainer.evaluate(q_network=q_network, n_episodes=20, eps=0.01)
    trainer.render(q_network=q_network, eps=0.01)
