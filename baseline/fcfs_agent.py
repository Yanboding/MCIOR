import numpy as np

from utils import getInsertionIndex


class Baseline:

    def __init__(self, threshold):
        self.threshold = threshold

    def _select_action(self, unboundedRemainingLifetimes, action_mask):
        unboundedRemainingLifetimes = np.array(unboundedRemainingLifetimes)
        indexs = sorted(range(len(unboundedRemainingLifetimes)), key=lambda k: unboundedRemainingLifetimes[k])
        i = getInsertionIndex(self.threshold, unboundedRemainingLifetimes[indexs])
        for j in range(i, len(indexs)):
            if action_mask[indexs[j]]:
                return indexs[j]
        for j in range(i - 1, -1, -1):
            if action_mask[indexs[j]]:
                return indexs[j]
        raise IndexError('No action is available')

    def select_action(self, state, action_mask):
        unboundedRemainingLifetimes = [state[3*i] for i in range(len(action_mask))]
        return self._select_action(unboundedRemainingLifetimes, action_mask)


if __name__ == '__main__':
    state = [101,60,70,90,100]
    action_mask = [True, True, True, True, False]
    threshold = 60
    baseline = Baseline(threshold)
    print(baseline._select_action(state, action_mask))
