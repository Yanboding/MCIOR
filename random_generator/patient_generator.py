import numpy as np
import pandas as pd
import scipy.stats as RNG
from scipy.special import gamma


class PatientGenerator:

    def __init__(self, classProbs, impatientAverages, imagingAverage, imagingStd, surgeryAverages, surgeryStds, seed):
        '''
        :param classProbs:
        :param impatientAverages:
        :param imagingAverage:
        :param imagingStd:
        :param surgeryAverages:
        :param surgeryStds:
        :param seed:
        '''
        self.classProbs = classProbs
        self.impatientAverages = impatientAverages
        self.imagingAverage = imagingAverage
        self.imagingStd = imagingStd
        self.surgeryAverages = surgeryAverages
        self.surgeryStds = surgeryStds

        self.classNumber = len(classProbs)

        self.seed = seed

        # 1. patient class generator
        self.patientClassGenerator = np.random.default_rng(seed)

        # 2. impatient time generators
        self.impatientTimeGenerators = []
        for i in range(len(impatientAverages)):
            impatientTimeGenerator = RNG.weibull_min(c=1.5, scale=impatientAverages[i]/gamma(1+1/1.5))
            impatientTimeGenerator.random_state = np.random.RandomState(seed=self.nextSeed())
            self.impatientTimeGenerators.append(impatientTimeGenerator)

        # 3. imaging time generator
        self.potentialImagingTimeGenerator = RNG.norm(loc=imagingAverage, scale=imagingStd)
        self.potentialImagingTimeGenerator.random_state = np.random.RandomState(seed=self.nextSeed())

        # 4. surgery time
        self.potentialSurgeryTimeGenerators = []
        for averageSurgeryTime, surgeryTimeStd in zip(surgeryAverages, surgeryStds):
            sigma = np.sqrt(np.log(1 + (surgeryTimeStd**2 / averageSurgeryTime**2)))
            potentialSurgeryTimeGenerator = RNG.lognorm(s=sigma,scale=np.exp(np.log(averageSurgeryTime) - sigma**2/2))
            potentialSurgeryTimeGenerator.random_state = np.random.RandomState(seed=self.nextSeed())
            self.potentialSurgeryTimeGenerators.append(potentialSurgeryTimeGenerator)

        self.impatientTime = np.vectorize(lambda patientClass: self.impatientTimeGenerators[patientClass].rvs(), otypes=[float])
        self.potentialSurgeryTime = np.vectorize(lambda patientClass: self.potentialSurgeryTimeGenerators[patientClass].rvs(), otypes=[float])

    def nextSeed(self):
        self.seed += 1
        return self.seed

    def _rvs(self, size):
        patientClasses = self.patientClassGenerator.choice(np.arange(self.classNumber), size, p=self.classProbs)
        interarrivalTimes = np.zeros(size)
        impatientTimes = self.impatientTime(patientClasses)
        potentialImagingTimes = self.potentialImagingTimeGenerator.rvs(size)
        potentialSurgeryTimes = self.potentialSurgeryTime(patientClasses)
        patientClasses += 1
        return np.array([patientClasses, interarrivalTimes, impatientTimes, potentialImagingTimes, potentialSurgeryTimes]).T

    def rvs(self, size):
        patients = pd.DataFrame(self._rvs(size), columns=['patientClass',
                                                          'interarrivalTime',
                                                          'impatientTime',
                                                          'potentialImagingTime',
                                                          'potentialSurgeryTime'])
        return patients.astype({'patientClass':int})

if __name__ == '__main__':
    params = {
        'classProbs': [0.3, 0.3, 0.4],
        'impatientAverages': [100, 300, 800],
        'imagingAverage': 12,
        'imagingStd': 3,
        'surgeryAverages': [103, 103, 103],
        'surgeryStds': [30, 30, 30],
        'seed': 0
    }
    averages = []
    classProbs = np.array(params['classProbs'])
    impatientAverages = np.array(params['impatientAverages'])
    print([np.sum(classProbs*impatientAverages)]+params['impatientAverages'])
    #patientGenerator = PatientGenerator(**params)
    #print(patientGenerator.rvs(50))
