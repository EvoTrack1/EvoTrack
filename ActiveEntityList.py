from EntityEvolutionList import EntityEvolutionList

class ActiveEntity:
    def __init__(self, node_id, EntityEvolutionList, time_window_id, label = 0):
        self.node_id = node_id  
        self.EntityEvolutionList = EntityEvolutionList  
        self.time_window_id = time_window_id  
        self.label = label

    def update_time_window_id(self,time_window_id):
        self.time_window_id = time_window_id

class ActiveEntityList:
    def __init__(self, EvolutionList_max_length, max_size):
        self.EvolutionList_max_length = EvolutionList_max_length
        self.max_size = max_size  
        self.acticeNodelsit = []
        self.nodeid_listindex = {}

    def add_node_state(self, node_id, time_window_id, Entity_state = None, label = None):

        if node_id in self.nodeid_listindex:
            self.acticeNodelsit[self.nodeid_listindex[node_id]].update_time_window_id(time_window_id)
            if Entity_state == None:
                self.acticeNodelsit[self.nodeid_listindex[node_id]].EntityEvolutionList.add_old()
            else:
                self.acticeNodelsit[self.nodeid_listindex[node_id]].EntityEvolutionList.add_new(time_window_id, Entity_state)

        else:
            _EntityEvolutionList = EntityEvolutionList(self.EvolutionList_max_length)
            _EntityEvolutionList.add_new(time_window_id, Entity_state)
            new_node = ActiveEntity(node_id, _EntityEvolutionList, time_window_id)

            if len(self.acticeNodelsit) == self.max_size:
                 self._remove_least_recent_node()

            self.acticeNodelsit.append(new_node)
            self.nodeid_listindex[node_id] = len(self.acticeNodelsit)-1

        if label == 1:
            self.acticeNodelsit[self.nodeid_listindex[node_id]].label = 1


    def _remove_least_recent_node(self):
        earliest_node = min(self.acticeNodelsit, key=lambda node: node.time_window_id)
        earliest_node_id = earliest_node.node_id
        del self.nodeid_listindex[earliest_node_id]
        remove_index = self.acticeNodelsit.index(earliest_node)
        self.acticeNodelsit.remove(earliest_node)
        for nodeid, listindex in self.nodeid_listindex.items():
            if listindex > remove_index:
                self.nodeid_listindex[nodeid] = self.nodeid_listindex[nodeid] - 1

    def get_Evolution(self, node_id):
        return self.acticeNodelsit[self.nodeid_listindex[node_id]]
    
    
    def get_acticeNodelsit(self):
        my_list = []
        for i in self.acticeNodelsit:
            my_list.append(i.node_id)
        return my_list




