from torch.utils.data import Dataset
import os

def replace_user_path(path):
    user_parents = [
        '/home',
        '/export/home',
        '/Users',
        'C:/Users',
        'C:/Documents and Settings'
    ]
    
    original_sep = '\\' if '\\' in path else '/'
    
    normalized_path = path.replace(original_sep, '/')
    
    for parent in user_parents:
        normalized_parent = parent.replace('\\', '/')
        prefix = f"{normalized_parent}/"
        
        if normalized_path.startswith(prefix):
            suffix = normalized_path[len(prefix):]
            parts = suffix.split('/')
            
            if not parts or not parts[0]:
                continue
            
            new_suffix = '*'
            if len(parts) > 1:
                new_suffix += '/' + '/'.join(parts[1:])
            
            new_path = f"{normalized_parent}/{new_suffix}".replace('/', original_sep)
            return new_path
        
        elif normalized_path == normalized_parent:
            return path
    
    return path


class EroTrackDataset(Dataset):
    def __init__(self, dataset_name):
        self. dataset = []
        self.dataset_name = dataset_name
        
        for subfolder in os.listdir('datasets/' + self.dataset_name):
            subfolder_path = os.path.join('datasets/' + self.dataset_name, subfolder)
    
            if os.path.isdir(subfolder_path):
                dataset_t = {}

                link_file_path = os.path.join(subfolder_path, 'link')
                node_file_path = os.path.join(subfolder_path, 'node')

                dataset_t['node'] = []
                with open(node_file_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        parts = line.strip().split(' ')
                        if len(parts) == 4:
                            dataset_t['node'].append([int(parts[0]), int(parts[1]), int(parts[2]), replace_user_path(parts[3])])
                        elif len(parts) == 3:
                            dataset_t['node'].append([int(parts[0]), int(parts[1]), int(parts[2]), 'None'])
                        else:
                            dataset_t['node'].append([int(parts[0]), int(parts[1]), int(parts[2]), parts[3]+ " " + parts[4]])

                dataset_t['link'] = []
                with open(link_file_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        parts = line.strip().split(' ')
                        dataset_t['link'].append([int(parts[0]), int(parts[1]), int(parts[2])])

                self.dataset.append(dataset_t)

        
    def __getitem__(self, idx):
        return self.dataset[idx]

    def get_time_windows_number(self):
        return  len(self.dataset)
    






