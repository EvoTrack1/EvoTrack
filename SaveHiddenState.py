class ActiveEntity:
    def __init__(self, node_id, hidden_state, time_window_id):
        self.node_id = node_id  
        self.hidden_state = hidden_state  
        self.time_window_id = time_window_id  

    def update_time_window_id(self,time_window_id):
        self.time_window_id = time_window_id

class ActiveEntityHiddenStateList:
    def __init__(self, max_size):
        self.max_size = max_size  
        self.acticeNodelsit = []
        self.nodeid_listindex = {}

    def add_node_state(self, node_id, time_window_id, hidden_state = None):

        if node_id in self.nodeid_listindex:
            self.acticeNodelsit[self.nodeid_listindex[node_id]].update_time_window_id(time_window_id)
            if hidden_state != None:
                self.acticeNodelsit[self.nodeid_listindex[node_id]].hidden_state = hidden_state
        else:
            new_node = ActiveEntity(node_id, hidden_state, time_window_id)

            if len(self.acticeNodelsit) == self.max_size:
                 self._remove_least_recent_node()

            self.acticeNodelsit.append(new_node)
            self.nodeid_listindex[node_id] = len(self.acticeNodelsit)-1

    def _remove_least_recent_node(self):
        earliest_node = min(self.acticeNodelsit, key=lambda node: node.time_window_id)
        earliest_node_id = earliest_node.node_id
        del self.nodeid_listindex[earliest_node_id]
        remove_index = self.acticeNodelsit.index(earliest_node)
        self.acticeNodelsit.remove(earliest_node)
        for nodeid, listindex in self.nodeid_listindex.items():
            if listindex > remove_index:
                self.nodeid_listindex[nodeid] = self.nodeid_listindex[nodeid] - 1

    def get_hiddenstate(self, node_id):
        if node_id in self.nodeid_listindex:
            return self.acticeNodelsit[self.nodeid_listindex[node_id]].hidden_state
        else:
            return None
    
    def get_acticeNodelsit(self):
        my_list = []
        for i in self.acticeNodelsit:
            my_list.append(i.node_id)
        return my_list
    
    def get_window_id(self, node_id):
        if node_id in self.nodeid_listindex:
            return self.acticeNodelsit[self.nodeid_listindex[node_id]].time_window_id
        else:
            return None



