import re

def LoadAbstractGraph(dataname):
    node = []
    type = []
    subject = []
    object = []
    with open("AbstractGraph\\" + dataname + "\\node", 'r') as nf:
        for line in nf:
            parts = re.split(r'\s+', line.strip())
            node.append(int(parts[0]))
            type.append(int(parts[1]))
    with open("AbstractGraph\\" + dataname + "\\edge", 'r') as ef:
        for line in ef:
            parts = re.split(r'\s+', line.strip())
            subject.append(int(parts[0]))
            object.append(int(parts[1]))
    
    return node, type, subject, object