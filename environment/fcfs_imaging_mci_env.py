import numpy as np
import gymnasium as gym

from random_generator import PatientGenerator
from utils import EventCalendar, getInsertionIndex, TimeStat, LinkedList, Node, EventNotice


class MCI:

    def __init__(self, patients):
        self.patients = patients
        self.numberOfPatients = len(patients)
        self.calendar = EventCalendar()
        self.abandonTimeStat = TimeStat()
        self.survivalRate = np.vectorize(self._survivalRateFun, otypes=[float])

    def _survivalRateFun(self, t):
        numberOfDeath = getInsertionIndex(t, self.abandonTimeStat.times)
        return (self.numberOfPatients - numberOfDeath) / self.numberOfPatients

    def recordDeath(self, patientId):
        self.recordLeaveTime(patientId)
        self.patients.at[patientId, 'isAbandon'] = True
        self.abandonTimeStat.record(self.patients.at[patientId, 'leaveTime'])

    def recordLeaveTime(self, patientId):
        self.patients.at[patientId, 'currentResource'] = None
        self.patients.at[patientId, 'node'] = None
        self.patients.at[patientId, 'leaveTime'] = self.calendar.clock

class FCFSImagingEnv(MCI):
    def __init__(self, patients, classProbs, impatientAverages, orRoomNumber, ctRoomNumber):
        super().__init__(patients)
        self.patients = patients
        self.classProbs = classProbs
        self.orRoomNumber = orRoomNumber
        self.ctRoomNumber = ctRoomNumber

        self.classNumber = len(classProbs) + 1
        self.impatientAverages = np.array([np.sum(np.array(classProbs) * np.array(impatientAverages))] + impatientAverages)

    def reset(self):
        # Track patients that are currently in surgery rooms
        self.busyORrooms = set()
        # Track patients that are currently in CT rooms
        self.busyCTRooms = set()
        # Track queue for each group
        self.queues = [LinkedList() for _ in range(self.classNumber)]
        # The patient is at which queue
        self.patients['arrivalTime'] = self.patients['interarrivalTime'].cumsum()
        self.patients['abandonTime'] = self.patients['arrivalTime'] + self.patients['impatientTime']
        self.patients['isAbandon'] = False
        self.patients['isUsedORroom'] = False
        self.patients['isUsedCTroom'] = False
        self.patients['leaveTime'] = np.NaN
        self.patients['node'] = None
        self.patients['currentResource'] = self.queues[0]
        self.patients['currentClass'] = 0
        for pid, patient in self.patients.iterrows():
            node = Node(pid)
            self.patients.at[pid, 'node'] = node
            self.queues[0].add_back(node)
            if patient['abandonTime'] < float('inf'):
                self.calendar.add(EventNotice('Abandon', patient['abandonTime'], node))
        state, _, _, action_mask = self._getState()
        return state, {'action_mask': action_mask}

    def get_patient(self, pid):
        return pid, self.patients.iloc[pid]

    def _getState(self):
        unboundedAverageRemainingLifetimes = self.impatientAverages - self.calendar.clock
        action_mask = np.array([False]*self.classNumber, dtype=bool)
        state = [0, 0, 0] * self.classNumber
        busyORNumber = len(self.busyORrooms)
        busyCTNumber = len(self.busyCTRooms)
        waitingPatientsNumber = 0
        for node in self.busyORrooms:
            pid, patient = self.get_patient(node.val)
            state[int(3 * patient['currentClass'] + 2)] += 1
        for i, queue in enumerate(self.queues):
            state[3 * i] = unboundedAverageRemainingLifetimes[i]
            queueSize = len(queue)
            state[3 * i + 1] = queueSize
            waitingPatientsNumber += queueSize
            if queueSize > 0:
                action_mask[i] = True

        patientsInSystem = busyORNumber + busyCTNumber + waitingPatientsNumber
        # state, if the state is terminal, if the state need action
        state = np.array(state + [busyCTNumber])
        is_done = patientsInSystem == 0
        need_action = busyORNumber < min(patientsInSystem - busyCTNumber, self.orRoomNumber)
        return state, is_done, need_action, action_mask

    def tryEnterCTRooms(self):
        while len(self.busyCTRooms) < min(len(self.queues[0]) + len(self.busyCTRooms), self.ctRoomNumber):
            node = self.queues[0].remove_front()
            self.busyCTRooms.add(node)
            pid, patient = self.get_patient(node.val)
            endOfImagingTime = self.calendar.clock + patient['potentialImagingTime']
            if patient['abandonTime'] > endOfImagingTime:
                self.calendar.add(EventNotice('EndOfImaging', endOfImagingTime, node))

            self.patients.at[pid, 'isUsedCTroom'] = True
            self.patients.at[pid, 'currentResource'] = self.busyCTRooms
            self.patients.at[pid, 'currentClass'] = 0

    def endOfImaging(self, node):
        if node not in self.busyCTRooms:
            raise KeyError('Patient is not in CT room')
        pid, patient = self.get_patient(node.val)
        self.busyCTRooms.remove(node)
        self.queues[int(patient['patientClass'])].add_back(node)
        self.patients.at[pid, 'currentResource'] = self.queues[int(patient['patientClass'])]
        self.patients.at[pid, 'currentClass'] = patient['patientClass']

    def abandon(self, node):
        pid, patient = self.get_patient(node.val)
        if patient['node'] is not None:
            self.patients.at[pid, 'currentResource'].remove(node)
            self.recordDeath(pid)
            return 1
        return 0

    def endOfSurgery(self, node):
        pid, patient = self.get_patient(node.val)
        self.patients.at[pid, 'currentResource'].remove(node)
        self.recordLeaveTime(pid)

    def enterSurgeryRooms(self, action):
        if len(self.queues[action]) == 0:
            raise IndexError('Invalid action: ' + str(action))
        while len(self.busyORrooms) < min(len(self.queues[action]) + len(self.busyORrooms), self.orRoomNumber):
            node = self.queues[action].remove_front()
            self.busyORrooms.add(node)

            pid, patient = self.get_patient(node.val)
            endOfSurgeryTime = self.calendar.clock + patient['potentialSurgeryTime']
            if patient['abandonTime'] > endOfSurgeryTime:
                self.calendar.add(EventNotice('EndOfSurgery', endOfSurgeryTime, node))

            self.patients.at[pid, 'isUsedORroom'] = True
            self.patients.at[pid, 'currentResource'] = self.busyORrooms

    def step(self, action):
        self.enterSurgeryRooms(action)
        # handle same timestamp event at the same time
        death = 0
        while len(self.calendar) > 0:
            buffer = self.calendar.saveRemove()
            for event in buffer['Abandon']:
                death += self.abandon(event.eventObject)
            for event in buffer['EndOfImaging']:
                self.endOfImaging(event.eventObject)
            for event in buffer['EndOfSurgery']:
                self.endOfSurgery(event.eventObject)
            self.tryEnterCTRooms()
            state, done, needAction, action_mask = self._getState()
            if needAction:
                return state, -death, done, False, {'action_mask': action_mask}
        return state, -death, done, False, {'action_mask': action_mask}

class FCFSImagingEnvWrapper:

    def __init__(self, classProbs, impatientAverages, imagingAverage, imagingStd, surgeryAverages, surgeryStds, patientNumber, orRoomNumber, ctRoomNumber, seed=0):
        self.classProbs = classProbs
        self.impatientAverages = impatientAverages
        self.imagingAverage = imagingAverage
        self.imagingStd = imagingStd
        self.surgeryAverages = surgeryAverages
        self.surgeryStds = surgeryStds
        self.patientNumber = patientNumber
        self.orRoomNumber = orRoomNumber
        self.ctRoomNumber = ctRoomNumber
        self.seed = seed
        self.patientGenerator = PatientGenerator(classProbs, impatientAverages, imagingAverage, imagingStd, surgeryAverages, surgeryStds, seed)

        # openai gym attributes
        classNumber = len(self.classProbs)+1
        # action will be the portfolio weights from 0 to 1 for each asset
        self.action_space = gym.spaces.Discrete(len(self.classProbs)+1)

        # get the observation space from the data min and max
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                shape=(classNumber*3 + 1, ),
                                                dtype=np.float32)

    def reset(self):
        patients = self.patientGenerator.rvs(self.patientNumber)
        classNumber = len(self.classProbs)
        self.env = FCFSImagingEnv(patients, self.classProbs, self.impatientAverages, self.orRoomNumber, self.ctRoomNumber)
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)


