import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
class EntityEvolutionList:
    class Node:
        def __init__(self, time, value=None):
            self.time = time
            self.value = value
            self.next = None
            self.number = 1

    def __init__(self, max_length):
        self.max_length = max_length  
        self.head = None  
        self.tail = None  
        self.size = 0  

    def add_new(self, time, value):
        if not self.head:
            self.head = self.Node(time, value)
            self.tail = self.head
            self.size += 1
        else:
            if self.size == self.max_length:
                self._remove_head()

            new_node = self.Node(time, value)
            self.tail.next = new_node
            self.tail = new_node
            self.size += 1

    def add_old(self):
        self.tail.number = self.tail.number + 1
        self.size += 1

    def _remove_head(self):
        if self.head:
            self.head = self.head.next
            self.size -= 1
            if self.head is None:  
                self.tail = None

    def to_list(self):
        result = []
        current = self.head
        result.append(current.value)
        current = current.next
        t = current.time
        while t:
            result.append(torch.zeros(840).to(device))
            t = t - 1
        while current:
            result.append(current.value)
            n = current.number
            while n > 1:
                result.append(torch.zeros(840).to(device))
                n = n - 1
            current = current.next
        l = len(result)
        while (2-l)>0:
            result.append(torch.zeros(840).to(device))
            l = l + 1
        return result


