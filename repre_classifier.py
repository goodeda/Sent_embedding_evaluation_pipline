import torch.nn as nn
import torch.optim as opt
import torch
# from sklearn.metrics import classification_report

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class FFNN_mltlyr(nn.Module):
    # Feel free to add whichever arguments you like here.
    def __init__(self, vector_size, drp_rt, n_classes):
        super(FFNN_mltlyr, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(vector_size, 128),
            nn.Dropout(p=drp_rt),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        pred = self.linear_relu_stack(x)
        return pred


class FFNN_siglyr(nn.Module):
    # Feel free to add whichever arguments you like here.
    def __init__(self, vector_size, drp_rt, n_classes):
        super(FFNN_siglyr, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(vector_size, 128),
            nn.Dropout(p=drp_rt),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        pred = self.linear_relu_stack(x)
        return pred


class EnClassifier:
    def __init__(self, ebd, label, LAYERS, LEARNING_RATE, DROPOUT, N_EPOCHS=3, train_batch_size=32):
        self.X = ebd.to(device)
        self.y = torch.LongTensor(label).to(device)
        self.md_lyr = LAYERS
        self.drop_rate = DROPOUT
        self.epoch = N_EPOCHS
        self.lr = LEARNING_RATE
        self.model = FFNN_siglyr(self.X.shape[1], self.drop_rate, 3).to(device) if self.md_lyr == "siglyr" else FFNN_mltlyr(self.X.shape[1], self.drop_rate, 3).to(device)# 3->label number
        self.loss_fuc = nn.CrossEntropyLoss()
        self.batch = train_batch_size
        self.optimizer = opt.Adam(self.model.parameters(), lr=self.lr)

    def save_ckpt(self, n_epoch):
        torch.save(self.model.state_dict(),
                   "./models/en_xmlr_{0}_feature{1}_classifier_lr{2}_drp{3}_epoch{4}.pt".format(self.md_lyr, self.X.shape[1], self.lr, self.drop_rate, n_epoch))

    def training(self):
        # input vector length and label types
        print("## training ## learning rate %s, drop out %s, model %s" % (self.lr, self.drop_rate, self.md_lyr))
        for epoch in range(self.epoch):
            total_loss, correct = 0, 0
            for i in range(int(len(self.X) / self.batch) + 1):
                minibatch = self.X[i * self.batch:(i + 1) * self.batch]
                true_y = self.y[i * self.batch:(i + 1) * self.batch]
                pred = self.model(minibatch)
                loss = self.loss_fuc(pred, true_y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                max_value, pred_label = torch.max(pred, 1)
                correct += (pred_label == true_y).sum()
                if i == 6200:
                    self.save_ckpt(epoch+0.5)
            print("epoch %s finished" % epoch)
            print("acc:", correct / len(self.X))
            self.save_ckpt(epoch)  # save model on each epoch


def select_model(devX, devy, language, config):
    devX, devy = devX.to(device), torch.LongTensor(devy).to(device)
    # print(devX.shape[1])
    batch_size = config["dev_batch_size"]
    for lyr in config["md_lyr"]:
        models_perf = dict()
        for epoch in config["epoch"]:
            for dropout in config["dropout"]:
                for lr in config["learning_rate"]:
                    correct = 0
                    pred_y = []
                    loaded_model = FFNN_siglyr(devX.shape[1], dropout, 3) if lyr == "siglyr" else FFNN_mltlyr(devX.shape[1], dropout, 3)
                    loaded_model.load_state_dict(
                        torch.load("./models/en_xmlr_{0}_feature{1}_classifier_lr{2}_drp{3}_epoch{4}.pt".format(lyr, devX.shape[1], lr, dropout, epoch)))
                    with torch.no_grad():
                        for i in range(int(len(devX) / batch_size) + 1):
                            input = devX[i * batch_size:(i + 1) * batch_size]
                            true_y = devy[i * batch_size:(i + 1) * batch_size]
                            outs = loaded_model(input)
                            max_value, pred_label = torch.max(outs, 1)
                            correct += (pred_label == true_y).sum()
                            pred_y.append(outs)
                    correct = correct.detach().numpy()
                    models_perf[(lr, dropout, epoch, lyr)] = correct * 100 / len(devX)  # accuracy in %
                # "(learning_rate, dropout, epoch)"
    # record the acc of hyper-para combinations and in case need for manually check
        with open("log/model_selection_log".format(language), "a", encoding="utf-8") as f:
            f.write(language + " " + lyr + ":\n")
            f.write("\n".join(["(learning rate, dropout rate, epoch, layers): %s\t||\taccuracy: %s" % (comb[0], comb[1]) for comb in
                               models_perf.items()]))
            f.write("\n")
    return sorted(models_perf.items(), key=lambda x: x[1], reverse=True)[0]


# def evaluate(testX, testy, lr, dropout, epoch):
#     pred_y = []
#     testX = testX.to(device)
#     loaded_model = torch.load("./models/en_classifier_lr{0}_drp{1}_epoch{2}.pt".format(lr, dropout, epoch))
#     with torch.no_grad():
#         for input in testX:
#             outs = loaded_model(input)
#             max_value, pred_label = torch.max(outs, 1)
#             pred_y.extend(pred_label)
#     eval_result = classification_report(testy, pred_y, labels=["entailment", "neutral", "contradiction"])
#     return eval_result
