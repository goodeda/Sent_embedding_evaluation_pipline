from datasets import load_dataset
import numpy as np
import os

languages = ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']
data_part = ["train", "validation", "test"]
lang_data = load_dataset("xnli", "all_languages")

# creating folders
folders = ["embeddings", "models", "data", "log"]
for folder in folders:
    if not os.path.exists(folder):
        os.mkdir(folder)


def download_data():
    for data_seg in data_part:
        for lang_index, lang in enumerate(languages):
            lang_sent = [pair["translation"][lang_index] for pair in lang_data[data_seg]["hypothesis"]]
            with open("data/{0}_hypothesis_{1}.txt".format(lang, data_seg), "w", encoding="utf-8") as f:
                f.write("\n".join(lang_sent))
            lang_sent = [pair[lang] for pair in lang_data[data_seg]["premise"]]
            with open("data/{0}_premise_{1}.txt".format(lang, data_seg), "w", encoding="utf-8") as f:
                f.write("\n".join(lang_sent))
            print("%s %s download finished" % (lang, data_seg))
        np.save("data/{}_label.npy".format(data_seg), lang_data[data_seg]["label"])

download_data()
