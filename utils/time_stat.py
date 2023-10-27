from utils import getInsertionIndex


class TimeStat:

    def __init__(self):
        self.times = []

    def record(self, time):
        if self.times == [] or self.times[-1] <= time:
            self.times.append(time)
        elif self.times[0] > time:
            self.times.insert(0, time)
        else:
            self.times.insert(getInsertionIndex(time, self.times), time)