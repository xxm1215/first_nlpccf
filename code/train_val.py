import copy
import torch
import numpy as np
from click.core import batch
from torch.utils.data import DataLoader
import clip
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel

import os
import pandas as pd
import torch
# import clip
import requests
# from transformers import ChineseCLIPProcessor，ChineseCLIPModel
from transformers import ChineseCLIPProcessor, ChineseCLIPModel

from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Configs
DEVICE = "cuda:0"
NUM_WORKER = 1
BATCH_SIZE = 64
LR = 1e-3
L2 = 0  # 1e-5
NUM_EPOCH = 100
embed_dim = 512
num_heads = 1


# 数据集
class FakeDataset(Dataset):
    def __init__(self, df, folder_path, chineseclip_model):
        self.df = df
        self.folder_path = folder_path
        self.image_ids = df['id'].tolist()
        self.chineseclip_model = chineseclip_model

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.df.iloc[idx]['label']
        image_id = self.df.iloc[idx]['id']
        filename = f"{image_id}.jpg"
        image = Image.open(os.path.join(self.folder_path, filename))

        image_features, text_features = self.chineseclip_model.encode(image, image_id)
        return image_features, text_features, label


class ChineseCLIPModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
        self.preprocess = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
        self.model.eval()

    def encode(self, image, image_id):
        image_input = self.preprocess(images=image, return_tensors="pt")
        image_features = self.model.get_image_features(**image_input)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)  # normalize

        text = self.df.loc[self.df['id'] == image_id, 'text']
        text_input = self.preprocess(text=text, padding=True, return_tensors="pt")
        text_features = self.model.get_text_features(**text_input)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        return image_features, text_features


# Usage example
# chineseclip_model = ChineseCLIPModel()
# fake_dataset = FakeDataset(df, folder_path, chineseclip_model)


class MultiheadAttentionModel(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiheadAttentionModel, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, text_feature, img_feature):
        text_feature = text_feature.unsqueeze(1)
        img_feature = img_feature.unsqueeze(1)

        text_attn_output, _ = self.multihead_attn(img_feature, text_feature, text_feature)
        img_attn_output, _ = self.multihead_attn(text_feature, img_feature, img_feature)

        text_attn_output = text_attn_output.squeeze(1)
        img_attn_output = img_attn_output.squeeze(1)

        features = torch.cat((img_attn_output, text_attn_output), dim=-1)
        return features


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(1024, 512)  # 第一个全连接层，输入大小为64，输出大小为64
        self.fc2 = nn.Linear(512, 2)  # 第二个全连接层，输入大小为64，输出大小为2

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 使用ReLU激活函数
        x = self.fc2(x)
        return x


def train():
    # ---  Load Config  ---
    device = torch.device(DEVICE)
    num_workers = NUM_WORKER
    batch_size = BATCH_SIZE
    lr = LR
    l2 = L2
    num_epoch = NUM_EPOCH

    # ---  Load Data  ---
    # folder_path = '/mnt/qust_521_big_2/xxm_clip/nlpccf/weibo/rumor_images/'
    # df1=pd.read_csv('/mnt/qust_521_big_2/xxm_clip/nlpccf/weibo/weibo/test_rumor.csv', encoding='ISO-8859-1')
    # df2= pd.read_csv('/mnt/qust_521_big_2/xxm_clip/nlpccf/weibo/weibo/test_rumor.csv', encoding='ISO-8859-1')

    folder_path = '/mnt/qust_521_big_2/xxm_clip/nlpccf/weibo/nonrumor_images/'
    df1 = pd.read_csv('/mnt/qust_521_big_2/xxm_clip/nlpccf/weibo/weibo/train_nonrumor_existing.csv',
                      encoding='ISO-8859-1')
    df2 = pd.read_csv('/mnt/qust_521_big_2/xxm_clip/nlpccf/weibo/weibo/test_rumor_existing.csv', encoding='ISO-8859-1')
    print("1")
    train_set = FakeDataset(df1, folder_path)
    print("2")
    test_set = FakeDataset(df2, folder_path)
    print("3")
    print()
    train_loader = DataLoader(
        train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    print("4")
    print(train_set)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    print("5")
    # ---  Build Model & Trainer  ---
    chineseclip_model = ChineseCLIPModel()
    multiheadAttentionModel = MultiheadAttentionModel(embed_dim, num_heads)
    multiheadAttentionModel.to(device)
    classifier = Classifier()
    classifier.to(device)
    loss_func_detection = torch.nn.CrossEntropyLoss()
    print("6")
    optim_task_detection = torch.optim.Adam(
        list(multiheadAttentionModel.parameters()) + list(classifier.parameters()), lr=lr, weight_decay=l2
    )
    # ---  Model Training  ---
    # loss_similarity_total = 0
    loss_detection_total = 0
    best_acc = 0
    print("7")
    for epoch in range(num_epoch):
        chineseclip_model.train()

        multiheadAttentionModel.train()
        classifier.train()

        corrects_pre_detection = 0

        loss_detection_total = 0

        detection_count = 0
        print("8")
        pre_label_detection = []
        eval_labels = []
        for batch in train_loader:
            image_features, text_features, label = batch

            text_features = text_features.to(device)
            image_features = image_features.to(device)
            label = label.to(device)

            # ---  TASK2 Detection  ---
            # text_features=
            output = multiheadAttentionModel(text_features, image_features)
            pre_detection = classifier(output)
            loss_detection = loss_func_detection(pre_detection, label)

            # scheduler.step(loss_detection)


            optim_task_detection.zero_grad()
            loss_detection.backward()
            optim_task_detection.step()
            eval_labels.extend(label.cpu().tolist())
            # eval_preds.extend(torch.argmax(outputs.logits, dim=-1).cpu().tolist())



            pre_label_detection.extend(torch.argmax(pre_detection,dim=-1).cpu.tolist())
            # pre_label_detection = pre_detection.argmax(1)
            corrects_pre_detection=classification_report(eval_labels, pre_label_detection,, output_dict=True,
                                           labels=[0, 1], target_names=["real", "false"],
                                           )
            # corrects_pre_detection += pre_label_detection.eq(label.view_as(pre_label_detection)).sum().item()

            # ---  Record  ---
            loss_detection_total += loss_detection.item() * 3697
            detection_count += 3697

        loss_detection_train = loss_detection_total / detection_count
        acc_detection_train = corrects_pre_detection / detection_count

        # ---  Test  ---
        acc_detection_test, loss_detection_test, cm_detection = test(
            multiheadAttentionModel, classifier, test_loader)

        # ---  Output  ---
        print('---  TASK1 Detection  ---')
        print(
            "EPOCH = %d \n acc_detection_train = %.3f \n acc_detection_test = %.3f \n  best_acc = %.3f \n loss_detection_train = %.3f \n loss_detection_test = %.3f \n" %
            (epoch + 1, acc_detection_train, acc_detection_test, best_acc, loss_detection_train, loss_detection_test)
        )

        print('---  TASK1 Detection Confusion Matrix  ---')
        print('{}\n'.format(cm_detection))


def test(multiheadAttentionModel, classifier, test_loader):
    multiheadAttentionModel.eval()
    classifier.eval()

    device = torch.device(DEVICE)
    loss_func_detection = torch.nn.CrossEntropyLoss()

    # similarity_count = 0
    detection_count = 0
    # loss_similarity_total = 0
    loss_detection_total = 0
    # similarity_label_all = []
    detection_label_all = []
    # similarity_pre_label_all = []
    detection_pre_label_all = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            image_features, text_features, label = batch
            text_features = batch['text_feature']
            image_features = batch['image_feature']
            label = batch['label']
            batch_size = 934

            text_features = text_features.to(device)
            image_features = image_features.to(device)
            label = label.to(device)

            # ---  TASK2 Detection  ---
            output = multiheadAttentionModel(text_features, image_features)
            pre_detection = classifier(output)
            loss_detection = loss_func_detection(pre_detection, label)

            pre_label_detection = pre_detection.argmax(1)

            # ---  Record  ---
            loss_detection_total += loss_detection.item() * 934
            detection_count += 934
            detection_pre_label_all.append(pre_label_detection.detach().cpu().numpy())
            detection_label_all.append(label.detach().cpu().numpy())

        loss_detection_test = loss_detection_total / detection_count

        detection_pre_label_all = np.concatenate(detection_pre_label_all, 0)
        detection_label_all = np.concatenate(detection_label_all, 0)

        acc_detection_test = accuracy_score(detection_pre_label_all, detection_label_all)
        cm_detection = confusion_matrix(detection_pre_label_all, detection_label_all)

    return acc_detection_test, loss_detection_test, cm_detection


if __name__ == "__main__":
    train()
