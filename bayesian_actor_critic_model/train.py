import time

import torch
from tqdm import tqdm
from os import path

from bayesian_actor_critic_model.bac import BAC
from utils import Memory, ZFilter, plot_total_reward

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)
weight_file = path.realpath(__file__).\
    replace('bayesian_actor_critic_model/train.py','bayesian_actor_critic_model/weights/')

def train(env, agent, epoch_num, max_time_steps, batch_size, replay_memory, svd_low_rank, state_coefficient, fisher_coefficient):
    running_state = ZFilter(env.observation_space.shape)
    total_rewards = []
    start_time = time.time()
    for e in tqdm(range(1, epoch_num+1)):
        prev_state, info = env.reset()
        prev_state = running_state(prev_state)
        total_reward = 0
        for time_step in range(max_time_steps):
            action = agent.select_action(prev_state)
            # observation, reward, terminated, truncated, info
            state, reward, done, truncated, info = env.step(action)
            state = running_state(state)
            replay_memory.add(prev_state, action, state, reward, done)
            # Train agent after collecting sufficient data
            if time_step % 5 == 0:
                agent.save(weight_file)
            prev_state = state
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

if __name__ == '__main__':
    # test the performance of bac in mountain
    # consider change agent architecture to only define update rule.
    import gymnasium as gym
    from models import ContinuousPolicy, Value

    env = gym.make('MountainCarContinuous-v0')
    agent_spec = {
        'state_dim': env.observation_space.shape,
        'action_dim': env.action_space.shape,
        'actor': ContinuousPolicy,
        'critic': Value,
        'discount': 0.995,
        'tau':0.97,
        'advantage_flag': True,
        'actor_args': {},
        'critic_args': {},
        'actor_lr': 3e-3,
        'critic_lr': 2e-2,
        'likelihood_noise_level': 1e-4
    }
    agent = BAC(**agent_spec)
    batch_size = 20
    max_time_steps = 1000
    memory = Memory(env.observation_space.shape, env.action_space.shape, batch_size * max_time_steps)
    train_spec = {
        'env': env,
        'agent': agent,
        'epoch_num': 20000,
        'max_time_steps': max_time_steps,
        'batch_size': batch_size,
        'replay_memory': memory,
        'svd_low_rank': 50,
        'state_coefficient': 1,
        'fisher_coefficient':5e-5
    }
    total_rewards = train(**train_spec)
    plot_total_reward(total_rewards)