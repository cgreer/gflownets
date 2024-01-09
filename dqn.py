from dataclasses import dataclass, field, asdict
import random
import time
from typing import (
    Any,
    Dict,
    Tuple,
)
from types import SimpleNamespace as Record
import uuid

import gymnasium as gym
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import wandb

enu = enumerate

Tensor = torch.Tensor


def get_device(gpu=True):
    if gpu:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device("cpu")


@dataclass
class LinearSchedule:
    '''
    lerp from :start to :end over :duration steps. Then stay at :end.
    '''
    start: float
    end: float
    duration: int # ok if float too

    def __call__(self, t: int):
        slope = (self.end - self.start) / self.duration
        return self.end if t > self.duration else self.start + slope * t


@dataclass
class Stage:
    name: str
    count: int = 0 # XXX: overflow?
    tot_elapsed: float = 0.0 # XXX: overflow?

    # Private
    st_time: float = None

    def split(self):
        if self.st_time is None:
            self.st_time = time.time()
        else:
            self.count += 1
            self.tot_elapsed += (time.time() - self.st_time)
            self.st_time = None

    def rate(self):
        if self.count <= 0:
            return 0.0
        return self.count / self.tot_elapsed


@dataclass
class StageTimer:
    '''
    stimer = StageTimer(1000)
    for _ in range(1000):
        stimer["op1"].split()
        op1()
        stimer["op1"].split()

        stimer["op2"].split()
        op2()
        stimer["op2"].split()

    ## Output (every 1000 iterations) ##
    op1:2.5s op2:3.4s
    '''
    N: int # Report in every :N calls

    # Private
    calls: int = field(init=False)
    stages: Dict[str, Stage] = field(init=False)

    def __post_init__(self):
        self.calls = 0
        self.stages = {}

    def __getitem__(self, key):
        # If it's been :N iterations then report in
        self.calls += 1
        if (self.calls % self.N) == 0 and (self.calls >= self.N):
            self.report()

        # Return stage
        if key not in self.stages:
            self.stages[key] = Stage(name=key)
        return self.stages[key]

    def report(self, oneline=False):
        s = ""
        just = max(len(x) for x in self.stages) + 8
        for i, stage in enu(self.stages.values()):
            if oneline:
                pad = "   " if i != 0 else ""
            else:
                pad = "\n" if i != 0 else ""
            s += pad + f"{stage.name.ljust(just)} {stage.rate():.3f}/s"
        print(s)


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
    wrappers: str
    render_mode: str = None

    _env: Any = field(init=False)

    def __post_init__(self):
        env = gym.make(self.id, render_mode=self.render_mode)
        env = self.apply_wrappers(env)
        env.action_space.seed(self.seed) # XXX: When have to call this?
        self._env = env

    def apply_wrappers(self, env):
        if self.wrappers == "atari":
            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)
            env = EpisodicLifeEnv(env)
            if "FIRE" in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env)
            env = ClipRewardEnv(env)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayScaleObservation(env)
            env = gym.wrappers.FrameStack(env, 4)
        return env

    @property
    def obs_shape(self):
        return self._env.observation_space.shape

    @property
    def obs_dtype(self):
        return self._env.observation_space.dtype

    def actions(self):
        '''Discrete dist that has .sample() method'''
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
    tot_steps: int = 0

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)

        # Transition
        s = Step(
            obs=self.obs, # s you came from (s) (before you reset!)
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
        self.tot_steps += 1
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
    def encoded_obs_size(self) -> int:
        return self.obs_encoding_size

    @property
    def encoded_action_size(self) -> int:
        return self.n_possible_actions

    def encode_obs(self, obs: np.ndarray, device) -> Tensor:
        return torch.tensor(np.array(obs), dtype=torch.float32, device=device)


@dataclass
class Monitor:
    project: str
    config: 'Config'
    run: str = None
    ema_a: float = 0.05
    ema_return: float = 0.0
    max_return: float = 0.0

    # private
    wandb_run: Any = field(init=False)

    def __post_init__(self):
        if self.run is None:
            run = f"{self.config.seed}"
        self.wandb_run = wandb.init(
            project=self.project,
            name=run,
            config={
                "training_config": asdict(self.config)
            },
            reinit=True, # Won't make sep runs w/out this...
        )

    def post_episode(self, episode: GymEpisode):
        self.ema_return = ((1 - self.ema_a) * self.ema_return)
        self.ema_return += self.ema_a * episode.tot_rewards
        self.max_return = max(self.max_return, episode.tot_rewards)
        if self.wandb_run:
            # XXX: Rate limit?
            wandb.log({"ep_returns": episode.tot_rewards})
            wandb.log({"ep_returns_max": self.max_return})
            wandb.log({"ep_len": episode.tot_steps})


@dataclass
class ReplayBuffer:
    capacity: int # n of FIFO entries to support
    obs_shape: Tuple[int]
    obs_dtype: Any
    codec: Any
    device: Any

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
        assert isinstance(self.obs_shape, tuple)

        self.obs = np.zeros((self.capacity,) + self.obs_shape, dtype=self.obs_dtype)
        self.action = np.zeros((self.capacity, 1), dtype=int)
        self.reward = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity,) + self.obs_shape, dtype=self.obs_dtype)
        self.term = np.zeros((self.capacity, 1), dtype=np.float32) # note how float

    def add(self, step: Step):
        assert step.obs.dtype == self.obs_dtype

        curs = self.cursor
        self.obs[curs] = step.obs
        self.action[curs] = step.a
        self.reward[curs] = step.r
        self.next_obs[curs] = step.next_obs
        self.term[curs] = step.term # bool will be converted to 0/1

        # Move cursor / update size
        self.size = min(self.size + 1, self.capacity)
        if self.cursor >= self.capacity - 1:
            self.cursor = 0
        else:
            self.cursor += 1

    def sample(self, n: int):
        '''Return a batch of on-device tensors, ready for action'''
        assert self.size > n
        ind = np.random.choice(a=self.size, size=n, replace=False)
        ind.sort() # :fingers-crossed: affects cache line?

        # Compose batch
        # - Each attribute will be an on-device tensor
        batch = Record()
        batch.obs = self.codec.encode_obs(self.obs[ind], device=self.device)
        batch.next_obs = self.codec.encode_obs(self.next_obs[ind], device=self.device)
        batch.reward = torch.as_tensor(self.reward[ind], device=self.device)
        batch.action = torch.as_tensor(self.action[ind], device=self.device)
        batch.term = torch.as_tensor(self.term[ind], device=self.device)
        return batch


def sync_network(net1, net2):
    '''Copy net1's weights to net2'''
    for net2_param, net1_param in zip(net2.parameters(), net1.parameters()):
        net2_param.data.copy_(net1_param.data)


class QNetworkMLP(nn.Module):
    '''
    Input:
        - Batch of flattened observations

    Outputs:
        - A Q(s, a) for each action
    '''
    def __init__(self, inp_size: int, out_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(inp_size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, out_size),
        )

    def forward(self, x):
        return self.network(x)

    def get_action(self, obs_enc: Tensor):
        # unsqueeze required to make batch of size 1 for convnets
        q_values = self(obs_enc.unsqueeze(0))
        action = torch.argmax(q_values).item()
        return action


class QNetworkConv(nn.Module):
    '''
    Input:
        - Batch of (Batch, Channel, Row, Col)
        - Assumes input values are uint8 pixel values

    Outputs:
        - A Q(s, a) for each action
    '''

    def __init__(self, inp_size: int, out_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, out_size),
        )

    def forward(self, x):
        return self.network(x / 255.0)

    def get_action(self, obs_enc: Tensor):
        # unsqueeze required to make batch of size 1 for convnets
        q_values = self(obs_enc.unsqueeze(0))
        action = torch.argmax(q_values).item()
        return action


def set_seeds(seed, determ=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = determ


@dataclass
class DQNTrainer:
    env: Any
    codec: Any
    config: 'Config'

    q_network: Any = field(init=False, default=None)
    device: Any = field(init=False)

    def __post_init__(self):
        self.q_network = None
        self.device = get_device(self.config.gpu)

    def init_networks(self):
        if self.config.network_type == "conv":
            Net = QNetworkConv
        elif self.config.network_type == "mlp":
            Net = QNetworkMLP
        else:
            raise KeyError()
        inp_size, out_size = self.codec.encoded_obs_size, self.codec.encoded_action_size
        q_network = Net(inp_size, out_size).to(self.device)
        target_network = Net(inp_size, out_size).to(self.device)
        target_network.load_state_dict(q_network.state_dict())
        return q_network, target_network

    def init_optimizer(self, q_network):
        return optim.Adam(
            q_network.parameters(),
            lr=self.config.learning_rate,
        )

    def train(self):
        print("\nTraining")
        env, config = self.env, self.config
        device = self.device

        # Handle determinism
        set_seeds(config.seed, config.torch_deterministic)

        # Initialize Networks (Q and Target)
        q_network, target_network = self.init_networks()

        # Create replay buffer
        print("..building buffer")
        buffer = ReplayBuffer(
            capacity=config.buffer_size,
            obs_shape=env.obs_shape,
            obs_dtype=env.obs_dtype,
            codec=self.codec,
            device=device,
        )

        # Monitor
        mon = Monitor(f"DQN-{env.id}", config)

        # Train
        print("..training")
        eps_sched = LinearSchedule(config.start_eps, config.end_eps, config.total_steps * config.explore_frac)
        optimizer = self.init_optimizer(q_network)
        episode = env.spawn()
        for step_i in (pbar := tqdm(range(config.total_steps))):
            # Collect experiences
            # - Choose an action to take (either eps explore or on-policy)
            # - Take action -> advance env state
            # - Add transition to replay buffer
            # - If episode is over, reset it.
            if random.random() < eps_sched(step_i):
                action = env.actions().sample()
            else:
                obs_enc = self.codec.encode_obs(episode.obs, device)
                action = q_network.get_action(obs_enc)
            step = episode.step(action)
            buffer.add(step)
            if episode.done:
                mon.post_episode(episode)
                episode = env.spawn()

            # Train networks
            # - Update Q Network
            #   - Sample a batch of transitions from replay buffer.
            #   - Compute loss and update weights
            # - Sync Target Network
            #   - Copy the q network weights to target network after
            #     some specified timesteps.
            if step_i > config.train_start and step_i % config.train_freq == 0:
                samples = buffer.sample(config.batch_size)

                with torch.no_grad():
                    target_max, _ = target_network(samples.next_obs).max(dim=1)
                    td_target = samples.reward.flatten() + config.gamma * target_max * (1 - samples.term.flatten())
                prev_val = q_network(samples.obs).gather(1, samples.action).squeeze()
                loss = F.mse_loss(td_target, prev_val)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step_i % config.target_update_freq == 0:
                    sync_network(q_network, target_network)

            # Report in at end of step
            pbar.set_postfix({"ema(R)": mon.ema_return, "max(R)": mon.max_return})

        # Set trainer q network
        self.q_network = q_network

    def save_checkpoint(self, name, q_network, optimizer=None):
        checkpoint = {
            'q_network': q_network.state_dict(),
            'optimizer': optimizer.state_dict() if optimizer else None,
        }
        if name is None:
            name = f"checkpoint-{uuid.uuid4()}"
        torch.save(checkpoint, f"{name}.pth")

    def load_checkpoint(self, name):
        checkpoint = torch.load(f"{name}.pth")
        q_network, _ = self.init_networks()
        q_network.load_state_dict(checkpoint['q_network'])

        optimizer = None
        if checkpoint['optimizer']:
            optimizer = self.init_optimizer(q_network)
            optimizer.load_state_dict(checkpoint['optimizer'])

        return (q_network, optimizer)

    def evaluate(self, q_network, n_episodes=10, eps=0.05):
        env = self.env

        episode = env.spawn()
        ep_rewards, max_r = 0.0, 0.0
        ep_i = 0
        while ep_i < n_episodes:
            if random.random() < eps:
                action = env.actions().sample()
            else:
                obs_enc = self.codec.encode_obs(episode.obs, self.device)
                action = q_network.get_action(obs_enc)
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

    def render(self, q_network, eps=0.05):
        # Get a renderable env
        env = GymEnv(
            id=self.env.id,
            seed=self.env.seed,
            wrappers=self.env.wrappers,
            render_mode="rgb_array", # key arg for rendering
        )

        episode = env.spawn()
        frames = []
        frames.append(env.render())
        while True:
            if random.random() < eps:
                action = env.actions().sample()
            else:
                obs_enc = self.codec.encode_obs(episode.obs, self.device)
                action = q_network.get_action(obs_enc)
            episode.step(action)
            frames.append(env.render())
            if episode.done:
                break

        Renderer.to_video(frames, env.id)


@dataclass
class Config:
    total_steps: int = 100_000
    train_start: int = 20_000
    train_freq: int = 10
    target_update_freq: int = 500
    batch_size: int = 64
    gamma: float = 0.99
    learning_rate: float = 2.5e-4
    start_eps: float = 1.0
    end_eps: float = 0.05
    explore_frac: float = 0.20
    buffer_size: int = 10_000
    gpu: bool = True # use gpu if it's available
    seed: int = 1
    torch_deterministic: bool = True
    network_type: str = None
    wrappers: str = None


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
        env_id = "CartPole-v1"
        env_id = "BreakoutNoFrameskip-v4"
        N = 100_000
        seed = 1

        env = gym.make(env_id)
        env.action_space.seed(seed)
        obs, _ = env.reset(seed=seed)

        print(f"\nEnv: {env_id}")

        print("\nObservation:")
        print("  observation_space:", env.observation_space)
        print("  shape:", env.observation_space.shape)
        print("  dtype:", env.observation_space.dtype)
        print("  low:", env.observation_space.low)
        print("  high:", env.observation_space.high)
        print("  sample:", env.observation_space.sample())

        print("\nAction:")
        print("  action_space:", env.action_space)
        print("  shape (typically empty):", env.action_space.shape)
        print("  size:", env.action_space.n)
        print("  dtype:", env.action_space.dtype)
        print("  sample:", env.action_space.sample())
        print("  start (typically 0):", env.action_space.start)

        print("\nRunning random episodes...")
        ep_r = 0.0
        max_step_r = 0.0
        max_ep_r = 0.0
        st_time = time.time()
        for _ in range(N):
            valid_actions = env.action_space
            action = valid_actions.sample()
            obs, r, term, trunc, info = env.step(action)
            ep_r += r
            max_step_r = max(r, max_step_r)
            if term or trunc:
                max_ep_r = max(ep_r, max_ep_r)
                ep_r = 0.0
                env.reset()
        elapsed = time.time() - st_time

        print("\nRun stats:")
        print("  steps:", N)
        print("  elapsed:", f"{elapsed:.2f}")
        print("  rate:", f"{N / elapsed:.1f}")
        print("  max(R_ep):", max_ep_r)
        print("  max(R_step):", max_step_r)

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

    def get_env(self, env_id, seed=1):
        if env_id == "CartPole-v1":
            wrappers = None
        elif env_id == "BreakoutNoFrameskip-v4":
            wrappers = "atari"
        else:
            raise KeyError()
        return GymEnv(id=env_id, seed=seed, wrappers=wrappers)

    def check_env(self):
        env_id = "CartPole-v1"
        # env_id = "BreakoutNoFrameskip-v4"
        env = self.get_env(env_id)
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
        start, end, dur = 1.0, 0.05, 50
        sched = LinearSchedule(start, end, dur)
        for t in range(100):
            param = sched(t)
            print(t, param)
        assert sched(0) == start
        assert start >= sched(25) >= end
        assert sched(51) == end
        assert sched(200) == end

    def check_codec(self):
        env_id = "CartPole-v1"
        env = self.get_env(env_id)
        codec = GymCodec(env)
        device = get_device()

        print(codec.encoded_obs_size)
        print(codec.encoded_action_size)

        episode = env.spawn()
        print(codec.encode_obs(episode.obs, device))

    def replay_buffer(self, capacity=100):
        env_id = "CartPole-v1"
        env = self.get_env(env_id)
        codec = GymCodec(env)
        buffer = ReplayBuffer(
            capacity=capacity,
            obs_shape=env.obs_shape,
            obs_dtype=env.obs_dtype,
            codec=codec,
            device=get_device(),
        )

        # Fill to over capacity
        episode = env.spawn()
        for _ in range(2 * capacity):
            action = env.actions().sample()
            step = episode.step(action)
            buffer.add(step)
            if episode.done:
                episode = env.spawn()

        return buffer

    def check_replay_buffer(self):
        buffer = self.replay_buffer(capacity=100_000)

        # Sample a batch
        samples = buffer.sample(4)
        print(samples)

    def check_mlp_net(self):
        # Single call
        # Vectorized call
        # Argmax results
        env_id = "CartPole-v1"
        env = self.get_env(env_id)
        codec = GymCodec(env)
        gamma = 0.99
        device = "cpu"
        inp_size, out_size = codec.encoded_obs_size, codec.encoded_action_size
        q_network = QNetworkMLP(inp_size, out_size).to(device)

        # Policy call during sampling
        episode = env.spawn()
        obs_enc = codec.encode_obs(episode.obs, device=device)
        obs_enc = obs_enc.unsqueeze(0)
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

    def check_conv_net(self):
        device = "cpu"
        batch_size = 4
        obs_shape = (4, 84, 84) # shape of obs
        inp_size, out_size = -1, 4 # doesn't matter, 4 actions

        # Make network
        q_network = QNetworkConv(inp_size, out_size).to(device)

        # Single call
        obs_enc = torch.randn(obs_shape).unsqueeze(0)
        q_vals = q_network(obs_enc)
        action = torch.argmax(q_vals).item()
        print(q_vals)
        print(action)

        # Batch call
        obs_enc = torch.randn((batch_size, *obs_shape))
        q_vals = q_network(obs_enc)
        max_qs, _ = q_vals.max(dim=1)
        print(obs_enc.shape)
        print(q_vals)
        print(max_qs)

    def check_stage_timer(self):
        unit = 0.001

        def op1():
            time.sleep(unit + int(random.random() * unit))

        def op2():
            time.sleep(3 * unit + int(random.random() * unit))

        stimer = StageTimer(20)
        for _ in range(1000):
            stimer["op1"].split()
            op1()
            stimer["op1"].split()

            stimer["op2"].split()
            op2()
            stimer["op2"].split()

    def gpu_bandwidth_test(self):
        device = get_device()

        # Multiple transfers
        print("\nMultiple")
        stimer = StageTimer(1000)
        for _ in range(10_000):
            stimer["op"].split()
            obs = torch.randn((2, 84, 84), dtype=torch.float).to(device)
            obs = torch.randn((2, 84, 84), dtype=torch.float).to(device)
            obs = torch.randn((2, 84, 84), dtype=torch.float).to(device)
            obs = torch.randn((2, 84, 84), dtype=torch.float).to(device)
            stimer["op"].split()

        # one transfer
        print("\nSingle")
        stimer = StageTimer(1000)
        for _ in range(10_000):
            stimer["op"].split()
            obs = torch.randn((8, 84, 84), dtype=torch.float).to(device) # noqa
            stimer["op"].split()


def train_cartpole():
    # Train
    env_id = "CartPole-v1"
    seed = random.randint(1, 1e6)
    config = Config(
        total_steps=150_000,
        train_start=10_000,
        train_freq=10,
        batch_size=128,
        learning_rate=2.5e-4,
        start_eps=1.00,
        end_eps=0.01,
        explore_frac=0.20,
        seed=seed,
        network_type="mlp",
        wrappers=None,
    )
    env = GymEnv(env_id, seed=seed, wrappers=config.wrappers)
    codec = GymCodec(env)
    trainer = DQNTrainer(env, codec, config)
    trainer.train()

    # Evaluate + Render video
    q_network = trainer.q_network
    trainer.evaluate(q_network=q_network, n_episodes=30, eps=0.01)
    trainer.render(q_network=q_network, eps=0.01)


def train_breakout():
    # Train
    env_id = "BreakoutNoFrameskip-v4"
    seed = random.randint(1, 1e6)
    total_steps = 2000
    config = Config(
        network_type="conv",
        total_steps=total_steps, # 10_000_000
        train_start=500, # 80_000
        train_freq=4,
        target_update_freq=1000,
        batch_size=32,
        learning_rate=1e-4,
        start_eps=1.00,
        end_eps=0.10, # 0.10
        explore_frac=0.10,
        buffer_size=1_000_000, # 1_000_000
        seed=seed,
        wrappers="atari",
    )
    env = GymEnv(env_id, seed=seed, wrappers=config.wrappers)
    codec = GymCodec(env)
    trainer = DQNTrainer(env, codec, config)
    trainer.train()

    # Save final model
    checkpoint_name = f"breakout_dqn-{seed}"
    trainer.save_checkpoint(checkpoint_name, trainer.q_network)

    # Evaluate
    q_network, _ = trainer.load_checkpoint(checkpoint_name)
    trainer.evaluate(q_network=q_network, n_episodes=30, eps=0.01)
    trainer.render(q_network=q_network, eps=0.01)


if __name__ == "__main__":
    Tasks().inspect_gym_env()
    Tasks().check_render()
    Tasks().check_env()
    Tasks().test_linear_sched()
    Tasks().check_codec()
    Tasks().check_replay_buffer()
    Tasks().check_mlp_net()
    Tasks().check_conv_net()
    Tasks().check_stage_timer()
    Tasks().gpu_bandwidth_test()
    import sys; sys.exit() # noqa

    for _ in range(3):
        train_cartpole()

    '''
    for _ in range(1):
        train_breakout()
    '''
