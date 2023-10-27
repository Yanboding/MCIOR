class Node:

    def __init__(self, val):
        self.val = val
        self.prev = None
        self.next = None

    def isOrphan(self):
        return self.prev == None and self.next == None

class LinkedList:

    def __init__(self, metaData=None):
        self.head = Node(None)  # Dummy head
        self.tail = Node(None)  # Dummy tail
        self.head.next = self.tail
        self.tail.prev = self.head
        self.metaData = metaData
        self.count = 0

    def is_empty(self):
        return self.head.next == self.tail

    def add_front(self, new_node):
        new_node.next = self.head.next
        new_node.prev = self.head
        self.head.next.prev = new_node
        self.head.next = new_node
        self.count += 1

    def add_back(self, new_node):
        new_node.next = self.tail
        new_node.prev = self.tail.prev
        self.tail.prev.next = new_node
        self.tail.prev = new_node
        self.count += 1

    def remove_front(self):
        if self.is_empty():
            return None
        front_node = self.head.next
        self.head.next = front_node.next
        front_node.next.prev = self.head
        front_node.next = front_node.prev = None
        self.count -= 1
        return front_node

    def remove_back(self):
        if self.is_empty():
            return None
        back_node = self.tail.prev
        self.tail.prev = back_node.prev
        back_node.prev.next = self.tail
        back_node.next = back_node.prev = None
        self.count -= 1
        return back_node

    def remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev
        node.next = node.prev = None
        self.count -= 1

    def __len__(self):
        return self.count

    def display(self):
        current = self.head.next
        while current != self.tail:
            print(current.val, end=' ')
            current = current.next
        print()