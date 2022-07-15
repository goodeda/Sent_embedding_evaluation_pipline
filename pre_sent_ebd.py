import numpy as np
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
model = AutoModel.from_pretrained("xlm-roberta-base")

print("reading file")
file = 'en_premise_train.txt'
with open(file, encoding="utf-8") as f:
    en_pre = f.read()
en_pre = en_pre.split("\n")
print(len(en_pre))

print("generating embeddings")
en_pre_representations = []
model.eval()
for i, sent in enumerate(en_pre):
    encoded_input = tokenizer(sent, return_tensors='pt')
    output = model(**encoded_input)
    en_pre_representations.append(output[1].detach())
    if i % 3000 == 0:
        print("3000 done")
print(len(en_pre_representations))
np.save("en_premise_train_xmlr_ebd.npy", en_pre_representations)