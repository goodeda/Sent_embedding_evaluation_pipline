import numpy as np
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
model = AutoModel.from_pretrained("xlm-roberta-base")
languages = ['zh'] #'ar', 'bg', 'de', 'el', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi',

for lang in languages:
    print("reading %s hypothesis file" % lang)
    file = 'data/%s_hypothesis_validation.txt' % lang
    print("reading %s" % file)
    with open(file, encoding="utf-8") as f:
        hyp = f.read()
    hyp = hyp.split("\n")
    print(len(hyp))

    print("generating hyp embeddings")
    hyp_representations = []
    model.eval()
    for i, sent in enumerate(hyp):
        encoded_input = tokenizer(sent, return_tensors='pt')
        output = model(**encoded_input)
        hyp_representations.append(output[1].detach())
    print(len(hyp_representations))
    np.save("embeddings/%s_hyp_validation.npy" % lang, hyp_representations)
    ##############################################################################
    print("reading %s premise file" % lang)
    file = 'data/%s_premise_validation.txt' % lang
    print("reading %s" % file)
    with open(file, encoding="utf-8") as f:
        pre = f.read()
    pre = pre.split("\n")
    print(len(pre))

    print("generating pre embeddings")
    pre_representations = []
    model.eval()
    for i, sent in enumerate(pre):
        encoded_input = tokenizer(sent, return_tensors='pt')
        output = model(**encoded_input)
        pre_representations.append(output[1].detach())
    print(len(pre_representations))
    np.save("embeddings/%s_pre_validation.npy" % lang, np.asarray(pre_representations))
