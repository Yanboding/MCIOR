import numpy as np
from tqdm import tqdm

from baseline import Baseline
from environment import FCFSImagingEnvWrapper
from utils import RunningStat
def run(env, agent, epoch_num, max_time_steps):
    stopTime = 2000
    t = np.linspace(0, stopTime, stopTime + 1)
    survivalRateStat = RunningStat(t.shape)
    total_rewards = []
    for e in tqdm(range(1, epoch_num+1)):
        prev_state, prev_info = env.reset()
        total_reward = 0
        for time_step in range(max_time_steps):
            action_mask = prev_info['action_mask'] if 'action_mask' in prev_info else np.array([True])
            action = agent.select_action(prev_state, action_mask)
            # observation, reward, terminated, truncated, info
            state, reward, done, truncated, info = env.step(action)
            prev_state, prev_info = state, info
            total_reward += reward
            if done or truncated:
                break
        survivalRateSamplePath = env.env.survivalRate(t)
        survivalRateStat.record(survivalRateSamplePath)
        total_rewards.append(total_reward)
    return {
        'survivalRateStat': survivalRateStat,
        'total_rewards': total_rewards
    }

params = {
        'patientNumber': 100,
        'classProbs': [0.01]*100,
        'impatientAverages': [100 + i for i in range(100)],
        'imagingAverage': 12,
        'imagingStd': 3,
        'surgeryAverages': [103]*100,
        'surgeryStds': [30]*100,
        'seed': 0,
        'orRoomNumber': 10,
        'ctRoomNumber': 0
    }
env = FCFSImagingEnvWrapper(**params)
agent = Baseline(103)

res = run(env, agent, 500, 100)
print(res['survivalRateStat'].mean()[-1], np.mean(res['total_rewards']))
