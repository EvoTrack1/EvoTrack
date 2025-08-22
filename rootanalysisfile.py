import csv

dataset_name = "theia"
filelist = ['ta1-theia-e3-official-6r.json.9',
 'ta1-theia-e3-official-1r.json.9',
 'ta1-theia-e3-official-6r.json.8',
 'ta1-theia-e3-official-6r.json.12',
 'ta1-theia-e3-official-1r.json.7',
 'ta1-theia-e3-official-6r.json.7',
 'ta1-theia-e3-official-6r.json.5',
 'ta1-theia-e3-official-1r.json.3',
 'ta1-theia-e3-official-6r.json',
 'ta1-theia-e3-official-1r.json.5',
 'ta1-theia-e3-official-6r.json.11',
 'ta1-theia-e3-official-1r.json.4',
 'ta1-theia-e3-official-1r.json.6',
 'ta1-theia-e3-official-5m.json',
 'ta1-theia-e3-official-1r.json.2',
 'ta1-theia-e3-official-6r.json.10',
 'ta1-theia-e3-official-6r.json.4',
 'ta1-theia-e3-official-6r.json.1',
 'ta1-theia-e3-official-3.json',
 'ta1-theia-e3-official-1r.json.8',
 'ta1-theia-e3-official-1r.json.1',
 'ta1-theia-e3-official-6r.json.6',
 'ta1-theia-e3-official-6r.json.2',
 'ta1-theia-e3-official-6r.json.3',
 'ta1-theia-e3-official-1r.json']


link = []

with open(dataset_name, 'r') as file:
    for line in file:
        split_line = line.strip().split('\t')
        link.append(split_line)

subject_to_object_events = [
    "EVENT_FORK", "EVENT_SIGNAL", "EVENT_EXECUTE", "EVENT_CREATE_OBJECT", "EVENT_CLONE",
    "EVENT_OPEN", "EVENT_CLOSE", "EVENT_WRITE", "EVENT_LSEEK", "EVENT_WRITE_SOCKET_PARAMS"
    "EVENT_FCNTL", "EVENT_MODIFY_FILE_ATTRIBUTES", "EVENT_TRUNCATE",
    "EVENT_RENAME", "EVENT_LINK", "EVENT_UNLINK", "EVENT_BIND",
    "EVENT_CONNECT", "EVENT_LISTEN", "EVENT_SENDMSG", "EVENT_SENDTO",
    "EVENT_FLOWS_TO"
]

object_to_subject_events = [
    "EVENT_READ", "EVENT_ACCEPT", "EVENT_RECVMSG", "EVENT_RECVFROM",
    "EVENT_MMAP", "EVENT_UNLINK", "EVENT_READ_SOCKET_PARAMS",
]


id = 0
uuid_id = {}
id_attr = {}

link_set = set()
link1 = []

for l in link:
    if l[0] not in uuid_id:
        uuid_id[l[0]] = id
        id_attr[id] = l[1]
        id = id + 1
    if l[3] not in uuid_id:
        uuid_id[l[3]] = id
        id_attr[id] = l[8]
        id  = id + 1
    
    if l[6] in subject_to_object_events:
        if str(uuid_id[l[0]]) + '_' + str(uuid_id[l[3]]) not in link_set:
            link1.append([uuid_id[l[0]], uuid_id[l[3]]])

    if l[6] in object_to_subject_events:
        if str(uuid_id[l[3]]) + '_' + str(uuid_id[l[0]]) not in link_set:
            link1.append([uuid_id[l[3]], uuid_id[l[0]]])

with open('Root\\node_attributes.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["node_id", "attribute"])
    for node_id, attr in id_attr.items():
        writer.writerow([node_id, attr])

with open('Root\\edge_list.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["source", "target"])
    for edge in link1:
        writer.writerow(edge)

