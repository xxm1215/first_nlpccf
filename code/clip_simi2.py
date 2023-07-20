import gc

from sklearn.metrics import classification_report
import copy
import random
import shutil
import torch
import numpy as np
from click.core import batch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import PIL
import os
import pandas as pd
import torch
import requests
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from transformers import CLIPProcessor

from PIL import Image

import numpy as np
from torchvision.transforms import ToTensor
from torchvision.transforms import ToTensor, Resize
from torchvision import transforms
import warnings

warnings.filterwarnings('ignore')

# Configs

DEVICE = "cuda:0"
NUM_WORKER = 1
BATCH_SIZE = 256
LR = 1e-3
L2 = 0  # 1e-5
NUM_EPOCH = 30
embed_dim = 512
num_heads = 1


# 数据集


class FakeDataset(Dataset):
    def __init__(self, df, folder_path):
        self.device = torch.device("cuda:0")
        self.df = df
        print(df)
        self.folder_path = folder_path
        self.image_files = None
        self.image_ids = df['id'].tolist()
        self.max_size = 224
        self.transform = transforms.Compose([

            transforms.Resize((self.max_size, self.max_size)),  # 调整图像大小为最大尺寸
            transforms.ToTensor()
        ])
        self.model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
        self.preprocess = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
        self.model = self.model.to(self.device)
        self.model.eval()

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]['text']
        label = self.df.iloc[idx]['label']
        image_id = self.df.iloc[idx]['id']

        if self.image_files is None:
            self.image_files = os.listdir(self.folder_path)
        filename = self.image_files[idx]
        image = Image.open(os.path.join(self.folder_path, filename)).convert("RGB")

        image = self.transform(image)

        image_input = self.preprocess(images=image, return_tensors="pt")
        image_input = image_input.to(self.device)
        image_features = self.model.get_image_features(**image_input)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)  # normalize
        text_input = self.preprocess(text=text, padding=True, return_tensors="pt")
        text_input = text_input.to(self.device)
        text_features = self.model.get_text_features(**text_input)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        text_features = text_features.detach()
        image_features = image_features.detach()

        image_features = image_features.to(self.device)
        text_features = text_features.to(self.device)

        return image_features, text_features, label

    def get_max_image_size(self):
        max_width = 0
        max_height = 0
        for image_id in self.image_ids:
            filename = f"{image_id}.jpg"
            image = Image.open(os.path.join(self.folder_path, filename))
            width, height = image.size
            max_width = max(max_width, width)
            max_height = max(max_height, height)
        return (max_width, max_height)


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiheadAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.fc1 = nn.Linear(embed_dim * 2, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, text_feature, img_feature):
        text_feature = text_feature.unsqueeze(1)
        img_feature = img_feature.unsqueeze(1)

        text_attn_output, _ = self.multihead_attn(img_feature, text_feature, text_feature)
        img_attn_output, _ = self.multihead_attn(text_feature, img_feature, img_feature)

        text_attn_output = text_attn_output.squeeze(1)
        img_attn_output = img_attn_output.squeeze(1)

        features = torch.cat((img_attn_output, text_attn_output), dim=-1)

        x = torch.relu(self.fc1(features))
        x = self.fc2(x)
        return x , text_attn_output,img_attn_output,features



def train():
    # ---  Load Config  ---
    device = torch.device("cuda:0")
    sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    sig = torch.nn.Sigmoid()
    num_workers = NUM_WORKER
    batch_size = BATCH_SIZE
    lr = LR
    l2 = L2
    num_epoch = NUM_EPOCH

    # ---  Load Data  ---

    folder_path = '/media/qust521/qust_3_big/fake_news_data/weibo/folder/images/'
    df1 = pd.read_csv('/media/qust521/qust_3_big/fake_news_data/weibo/nlpccf/weibo/train.csv')
    df2 = pd.read_csv('/media/qust521/qust_3_big/fake_news_data/weibo/nlpccf/weibo/test.csv')

    train_set = FakeDataset(df1, folder_path)


    test_set = FakeDataset(df2, folder_path)


    train_loader = DataLoader(
        train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True    )

    test_loader = DataLoader(
        test_set, batch_size=batch_size, num_workers=num_workers, shuffle=False    )
    # ---  Build Model & Trainer  ---


    multiheadAttentionModel = MultiheadAttention(embed_dim, num_heads)
    multiheadAttentionModel.to(device)
    loss_func_detection = torch.nn.CrossEntropyLoss()
    optim_task_detection = torch.optim.Adam(
        list(multiheadAttentionModel.parameters()),
        lr=lr, weight_decay=l2
    )
    # ---  Model Training  ---

    test_F1 = 0
    tolerant = 5
    saved_epoch = 0
    no_improvement_count = 0
    test_f1 = []
    best_test_accuracy = 0.0


    for epoch in range(num_epoch):
        multiheadAttentionModel.train()
        loss_detection_total = 0
        pre_label_detection = []
        labels = []
        c = 0
        sim_all = []
        mini_batch =[]
        for idx, batch in enumerate(train_loader):
            image = batch[0]
            text = batch[1]
            label = batch[2]
            # inputs = batch[3]

            label = label.type(torch.LongTensor)
            label = label.to(device)
            # train_loader.
            text_features = torch.squeeze(text)  # (64,1,512)-->(64,512)
            image_features = torch.squeeze(image)
            text_features = text_features.to(device)
            image_features = image_features.to(device)
            pre_detection,t_features,i_features,features = multiheadAttentionModel(text_features, image_features)


            sim_all.append(sim(t_features,i_features))
            mini_batch.append(features)
            sim_all = torch.stack(sim_all).to(device).squeeze()
            mini_batch = torch.stack(mini_batch).to(device).squeeze()
            sim_mean = torch.mean(sim_all)
            sim_std = torch.std(sim_all)
            normalized_mini = sig((mini_batch - sim_mean) / sim_std)
            pre_detection = normalized_mini * mini_batch

            loss_detection = loss_func_detection(pre_detection, label)

            optim_task_detection.zero_grad()
            loss_detection.backward()
            optim_task_detection.step()

            label = label.cpu().tolist()
            pre_detection = torch.argmax(pre_detection, dim=-1).cpu().tolist()

            train_loss = loss_detection.item()
            print("train iter: {}, train loss: {}".format(c + 1, train_loss))
            loss_detection_total += train_loss
            labels.extend(label)
            pre_label_detection.extend(pre_detection)
            c += 1
        loss_detection_total /= c
        print("train_total_loss: ", loss_detection_total)
        train_report = classification_report(labels, pre_label_detection, output_dict=True,
                                             target_names=["false", "true"])

        # ---  Test  ---
        loss_detection_test, test_report = test(
            multiheadAttentionModel, test_loader)
        test_accuracy = test_report['accuracy']
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_model_state = multiheadAttentionModel.state_dict()
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        print("Epoch: {}, Test Accuracy: {:.4f}".format(epoch + 1, test_accuracy))

        performance_metrics_1 = test_report["false"]
        df_1 = pd.DataFrame(performance_metrics_1,index=['false'])
        df_1.to_csv("/media/qust521/qust_3_big/fake_news_data/weibo/nlpccf/weibo/data_simi/test_false.csv", index=False)
        performance_metrics_2 = test_report["true"]
        df_2 = pd.DataFrame(performance_metrics_2,index=['true'])
        df_2.to_csv("/media/qust521/qust_3_big/fake_news_data/weibo/nlpccf/weibo/data_simi/test_true.csv", index=False)

        f1_macro = test_report['macro avg']['f1-score']
        print("macro F1 is: ", f1_macro)
        test_f1.append(f1_macro)
        if f1_macro > test_F1:
            test_F1 = f1_macro
            saved_model = multiheadAttentionModel
            saved_epoch = epoch
            print("saving model with test_f1 {} at Epoch {}".format(test_F1, saved_epoch + 1))

        torch.save(saved_model, "/media/qust521/qust_3_big/fake_news_data/weibo/nlpccf/weibo/data_simi/best_model.pth")


        if epoch-saved_epoch >= tolerant:
            multiheadAttentionModel.cpu()
            model = multiheadAttentionModel
            saved_model.cpu()


            del model, saved_model,pre_detection, loss_detection, labels, train_loader, test_loader, loss_detection_total
            gc.collect()
            torch.cuda.empty_cache()
            max_F1 =max(test_f1)
            with open("./test_finetune_t1.txt", 'w') as outf:
                outf.write("Best F1 score: {} at epoch {}\n".format(max_F1, epoch + 1))

            print("Early stopping at epoch {}.".format(epoch + 1))
            break

        # ---  Output  ---


def test(multiheadAttentionModel, test_loader):
    multiheadAttentionModel.eval()
    device = torch.device(DEVICE)
    loss_func_detection = torch.nn.CrossEntropyLoss()
    loss_detection_total = 0
    pre_label_detection = []
    labels = []
    c = 0
    sim_all = []
    mini_batch = []
    sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    sig = torch.nn.Sigmoid()

    with torch.no_grad():
        for batch in test_loader:
            image = batch[0]
            text = batch[1]
            label = batch[2]
            inputs = batch[3]

            label = label.type(torch.LongTensor)
            label = label.to(device)

            text_features = torch.squeeze(text)  # (64,1,512)-->(64,512)
            image_features = torch.squeeze(image)
            text_features = text_features.to(device)
            image_features = image_features.to(device)

            # ---  TASK2 Detection  ---
            pre_detection, t_features, i_features, features = multiheadAttentionModel(text_features, image_features)

            sim_all.append(sim(t_features, i_features))
            mini_batch.append(features)
            sim_all = torch.stack(sim_all).to(device).squeeze()
            mini_batch = torch.stack(mini_batch).to(device).squeeze()
            sim_mean = torch.mean(sim_all)
            sim_std = torch.std(sim_all)
            normalized_mini = sig((mini_batch - sim_mean) / sim_std)
            pre_detection = normalized_mini * mini_batch

            loss_detection = loss_func_detection(pre_detection, label)
            label = label.cpu().tolist()
            pre_detection = torch.argmax(pre_detection, dim=-1).cpu().tolist()

            test_loss = loss_detection.item()
            print("test iter: {}, test loss: {}".format(c + 1, test_loss))
            loss_detection_total += test_loss

            labels.extend(label)
            pre_label_detection.extend(pre_detection)
            c += 1
        loss_detection_total /= c
        loss_detection_test = loss_detection_total
        print("test_total_loss: ", loss_detection_total)
        test_report = classification_report(labels, pre_label_detection, output_dict=True,
                                            target_names=["false", "true"])

    return loss_detection_test, test_report


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    train()
