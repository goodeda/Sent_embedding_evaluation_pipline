from vector_prep import load_all_data
from repre_classifier import EnClassifier, select_model#, evaluate
from collections import defaultdict
import numpy as np

IFTEST = True
LAYER = ["siglyr", "mltlyr"]
languages = ['en', 'ar', 'bg', 'de', 'el', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']
Learning_rate = [5e-5, 5e-4]
Dropout = [0.1]
Dev_batch_size = 16
Max_epoch_times = 10
epochs = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5]
hyper_config = {"dev_batch_size": Dev_batch_size, "epoch": epochs, "learning_rate": Learning_rate,
                "dropout": Dropout, "md_lyr": LAYER}

# training English classifier
# all_en_data = load_all_data("en", IFTEST)
# for lr in Learning_rate:
#     for drp in Dropout:
#         classifier = EnClassifier(all_en_data["train_ebd"], all_en_data["train_label"], LAYER, lr, drp,
#                                   N_EPOCHS=Max_epoch_times, train_batch_size=32)
#         classifier.training()

# select hyper-parameters on each language dataset
parameter_selection = defaultdict(dict)
for lang in languages[1:]:  # exclude English
    lang_data = load_all_data(lang, IFTEST)
    selection_result = select_model(lang_data["validation_ebd"], lang_data["validation_label"], lang, hyper_config)
    parameter_selection[lang]["learning_rate"], parameter_selection[lang]["dropout"], \
    parameter_selection[lang]["epoch"], parameter_selection[lang]["layer"] = selection_result[0]
    parameter_selection[lang]["accuracy"] = selection_result[1]
# {"zh":{learning_rate:xxx, dropout:xxx, epoch:xxx, accuracy:xxx}}
print(parameter_selection)
with open("selection_output.txt", "a", encoding="utf-8") as f:
    f.write(" " + "input feature: %s" % lang_data["validation_ebd"].shape[1] + "\n")
    for language in parameter_selection:
        f.write(language+"\n")
        f.write(str(parameter_selection[language])+"\n")

# evaluation on test dataset
# for lang in languages[1:]:
#     with open("evaluation_outcome.txt", "a") as file:
#         result = evaluate(all_lang_data[lang]["test_ebd"], all_lang_data[lang]["test_label"], parameter_selection[lang]["learning_rate"],
#                  parameter_selection[lang]["dropout"], parameter_selection[lang]["epoch"])
#         file.write(result)
