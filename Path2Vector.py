from gensim.models import FastText
from dataset import EroTrackDataset
import os


dataname = ' '

dataset = EroTrackDataset(dataname)          
nodeattr  = set()
T =  100
for t in range(T):
    attr = [row[3] for row in dataset[t]['node']]
    for a in attr:
        nodeattr.add(a)
nodeattr = list(nodeattr)


model = FastText(nodeattr, vector_size=100, window=3, min_count=1, workers=4)

model_dir = "saved_models"
os.makedirs(model_dir, exist_ok=True)
model.save(os.path.join(model_dir, "fasttext_model.model"))

loaded_model = FastText.load(os.path.join(model_dir, "fasttext_model.model"))

