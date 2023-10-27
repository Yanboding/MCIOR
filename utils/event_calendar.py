from collections import defaultdict

from utils import getInsertionIndex


class EventNotice:

    def __init__(self, eventType, eventTime, eventObject):
        self.eventType = eventType
        self.eventTime = eventTime
        self.eventObject = eventObject

class EventCalendar:

    def __init__(self):
        self.clock = 0
        self.calendar = []

    def add(self, addedEvent):
        self.calendar.insert(getInsertionIndex(addedEvent, self.calendar, key=lambda x: x.eventTime),addedEvent)

    def remove(self):
        if len(self.calendar) > 0:
            nextEvent = self.calendar.pop(0)
            self.clock = nextEvent.eventTime
            return nextEvent

    def saveRemove(self):
        buffer = defaultdict(list)
        if len(self.calendar) > 0:
            nextEvent = self.remove()
            buffer[nextEvent.eventType].append(nextEvent)
            while len(self.calendar) > 0 and self.calendar[0].eventTime == self.clock:
                nextEvent = self.remove()
                buffer[nextEvent.eventType].append(nextEvent)
        return buffer

    def __len__(self):
        return len(self.calendar)