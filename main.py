import time
from os import path

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

from bayesian_actor_critic_model.bac import BAC
from environment import FCFSImagingEnvWrapper
from utils import ZFilter, Memory, plot_total_reward, RunningStat

torch.manual_seed(0)
torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)
weight_file = path.realpath(__file__).replace('main.py','bayesian_actor_critic_model/weights/')

def train(env, agent, epoch_num, max_time_steps, batch_size, replay_memory, svd_low_rank, state_coefficient, fisher_coefficient):
    running_state = ZFilter(env.observation_space.shape)
    total_rewards = []
    start_time = time.time()
    for e in tqdm(range(1, epoch_num+1)):
        prev_state, prev_info = env.reset()
        prev_state = running_state(prev_state)
        total_reward = 0
        for time_step in range(max_time_steps):
            action_mask = prev_info['action_mask'] if 'action_mask' in prev_info else np.array([True])
            action = agent.select_action(prev_state, action_mask)
            # observation, reward, terminated, truncated, info
            state, reward, done, truncated, info = env.step(action)
            state = running_state(state)
            replay_memory.add(prev_state, action, action_mask, state, reward, done)
            # Train agent after collecting sufficient data
            if time_step % 5 == 0:
                agent.save(weight_file)
            prev_state, prev_info = state, info
            total_reward += reward
            if done or truncated:
                break
        if e % batch_size == 0:
            agent.update(replay_memory, svd_low_rank, state_coefficient, fisher_coefficient)
            replay_memory.reset()
        total_rewards.append(total_reward)
        print('Iteration {:4d} - reward {:.3f} - Time elapsed: {:3f}sec'.
              format(e, total_reward, time.time() - start_time))
    return total_rewards

def evaluation(env, agent, max_time_steps, eval_times, episode_length):
    performance = RunningStat(episode_length)
    total_rewards = []
    for _ in range(eval_times):
        prev_state, info = env.reset()
        portfolio_value = []
        total_reward = 0
        for time_step in range(max_time_steps):
            action = agent.select_action(prev_state)
            # observation, reward, terminated, truncated, info
            state, reward, done, truncated, info = env.step(action)
            portfolio_value.append(info['portfolio_value'])
            prev_state = state
            total_reward += reward
            if done or truncated:
                break
        performance.record(np.array(portfolio_value))
        total_rewards.append(total_reward)
    return {'performance': performance,
            'total_rewards': total_rewards}

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    from models import DiscretePolicy, Value
    params = {
        'patientNumber': 50,
        'classProbs': [0.5, 0.5],
        'impatientAverages': [200, 450],
        'imagingAverage': 12,
        'imagingStd': 3,
        'surgeryAverages': [103, 103, 103],
        'surgeryStds': [30, 30, 30],
        'seed': 0,
        'orRoomNumber': 3,
        'ctRoomNumber': 2
    }
    env = FCFSImagingEnvWrapper(**params)
    # env = gym.make('MountainCar-v0')

    agent_spec = {
        'state_dim': env.observation_space.shape,
        'action_dim': env.action_space.shape,
        'actor': DiscretePolicy,
        'critic': Value,
        'discount': 0.995,
        'tau': 0.97,
        'advantage_flag': False,
        'actor_args': {'action_num': env.action_space.n},
        'critic_args': {},
        'actor_lr': 3e-3,
        'critic_lr': 2e-2,
        'likelihood_noise_level': 1e-4
    }
    agent = BAC(**agent_spec)
    batch_size = 150
    max_time_steps = 100
    memory = Memory(env.observation_space.shape, env.action_space.shape, env.action_space.n, batch_size * max_time_steps)
    train_spec = {
        'env': env,
        'agent': agent,
        'epoch_num': 30000,
        'max_time_steps': max_time_steps,
        'batch_size': batch_size,
        'replay_memory': memory,
        'svd_low_rank': 50,
        'state_coefficient': 1,
        'fisher_coefficient': 5e-5
    }
    total_rewards = train(**train_spec)
    plot_total_reward(total_rewards)
