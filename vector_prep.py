import numpy as np
import torch
from collections import defaultdict

languages = ['en', 'ar', 'bg', 'de', 'el', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']


def load_vector(file, test):
    all_rep = torch.Tensor()
    if not test:
        repre = np.load(file, allow_pickle=True)
        # each sentence embedding shape should be (50, 1, 512), 30 or 28 is batch size
        for batch in repre:
            all_rep = torch.cat((all_rep, batch), dim=1)  # add all sentence representations in one
        avg_representation = torch.mean(all_rep, dim=0)  # average 50 heads
        return avg_representation  # final shape should be (total_sent_num, 512[feature_dim])
    else:  # test vectors from multilingual BERT, with dim of (total_sent_num, 768[feature_dim])
        repre = np.load(file, allow_pickle=True)
        return torch.cat([i for i in repre])


def vectors2feature(vector1, vector2):
    # [v,u,|v-u|,v*u]
    '''
    the concatenated vectors should be 4 times as long as the original embeddings
    and shape is (sent_num, feature_dim)
    '''
    return torch.cat([vector1, vector2, np.abs(vector1-vector2), np.multiply(vector1, vector2)], axis=1) #


def load_all_data(lang, iftest):
    lang_data = defaultdict(dict)
    for split in ["train", "validation", "test"]:
        sent_ebd1 = load_vector("embeddings/{0}_hyp_{1}.npy".format(lang, split), iftest)
        sent_ebd2 = load_vector("embeddings/{0}_pre_{1}.npy".format(lang, split), iftest)
        # print(sent_ebd1.shape, sent_ebd2.shape)
        input_feature = vectors2feature(sent_ebd1, sent_ebd2)
        print(input_feature.shape)
        input_label = np.load("data/{}_label.npy".format(split))
        lang_data[split + "_ebd"] = input_feature
        lang_data[split + "_label"] = input_label
        print("%s loading done" % split)
    return lang_data
