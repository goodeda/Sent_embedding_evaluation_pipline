import numpy as np
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
model = AutoModel.from_pretrained("xlm-roberta-base")

print("reading file")
file = 'en_hypothesis_train.txt'
with open(file, encoding="utf-8") as f:
    en_hyp = f.read()
en_hyp = en_hyp.split("\n")
print(len(en_hyp))

print("generating embeddings")
en_hyp_representations = []
model.eval()
for i, sent in enumerate(en_hyp):
    encoded_input = tokenizer(sent, return_tensors='pt')
    output = model(**encoded_input)
    en_hyp_representations.append(output[1].detach())
    if i % 3000 == 0:
        print("3000 done")
print(len(en_hyp_representations))
np.save("en_hyp_train_xmlr_ebd.npy", en_hyp_representations)
