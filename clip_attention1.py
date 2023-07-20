from sklearn.metrics import classification_report
import copy
import random
import shutil
import torch
import numpy as np
from click.core import batch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import pandas as pd
import torch

from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
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
#DEVICE = "cpu"
NUM_WORKER = 1
BATCH_SIZE = 256
LR = 1e-3
L2 = 0  # 1e-5
NUM_EPOCH = 50
embed_dim = 512
num_heads = 1

#数据集



class FakeDataset(Dataset):
    def __init__(self, df, folder_path):
        #self.device = "cpu"
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
        #self.preprocess = self.preprocess.to(self.device)
        self.model.eval()
        #self.preprocess.eval()

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):
        # print(idx)
        # print(self.df)
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
        return x



def train():
# ---  Load Config  ---
    device = torch.device("cuda:0")
    #device = torch.device("cpu")
    num_workers = NUM_WORKER
    batch_size = BATCH_SIZE
    lr = LR
    l2 = L2
    num_epoch = NUM_EPOCH

    # ---  Load Data  ---

    
    folder_path1 = '/media/qust521/qust_3_big/fake_news_data/weibo/folder/nonrumor_images/'
    folder_path2 = '/media/qust521/qust_3_big/fake_news_data/weibo/folder/rumor_images/'
    folder_path = '/media/qust521/qust_3_big/fake_news_data/weibo/folder/images/'
    os.makedirs(folder_path,exist_ok=True)
    for filename in os.listdir(folder_path1):
        file_path1 = os.path.join(folder_path1,filename)
        file_path = os.path.join(folder_path, filename)
        shutil.copyfile(file_path1,file_path)
    for filename in os.listdir(folder_path2):
        file_path2 = os.path.join(folder_path2,filename)
        file_path = os.path.join(folder_path, filename)
        shutil.copyfile(file_path2,file_path)
    file_list = os.listdir(folder_path)
    random.shuffle(file_list)



        # folder_path = folder_path1+folder_path2


    df1_1 = pd.read_csv('/media/qust521/qust_3_big/fake_news_data/weibo/nlpccf/weibo/train_rumor_existing.csv',
                  names=["label", "id", "text"], header=None)
    df1_2 = pd.read_csv('/media/qust521/qust_3_big/fake_news_data/weibo/nlpccf/weibo/train_nonrumor_existing.csv',names=["label","id","text"],header=None)

    df1 = pd.concat([df1_1, df1_2], ignore_index=True)
    df1 = df1.sample(frac=1, random_state=42)
    df1.to_csv('/media/qust521/qust_3_big/fake_news_data/weibo/nlpccf/weibo/train.csv', index=False)
    nan_rows = df1['label'].isnull()
    df1 = df1[~nan_rows]


    df2_1 = pd.read_csv('/media/qust521/qust_3_big/fake_news_data/weibo/nlpccf/weibo/test_rumor_existing.csv',names=["label","id","text"],header=None)
    df2_2 = pd.read_csv('/media/qust521/qust_3_big/fake_news_data/weibo/nlpccf/weibo/test_nonrumor_existing.csv',names=["label","id","text"],header=None)
    df2 = pd.concat([df2_1, df2_2], ignore_index=True)
    df2 = df2.sample(frac=1, random_state=42)
    df2.to_csv('/media/qust521/qust_3_big/fake_news_data/weibo/nlpccf/weibo/test.csv', index=False)

    train_set = FakeDataset(df1, folder_path)
    # print("df2",df2)
    test_set = FakeDataset(df2, folder_path)
    # print()


    train_loader = DataLoader(
        train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True,#collate_fn=collate_fn1
    )
    # print(train_set)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, num_workers=num_workers, shuffle=False,#collate_fn=collate_fn1
    )
    # ---  Build Model & Trainer  ---



    multiheadAttentionModel = MultiheadAttention(embed_dim, num_heads)
    multiheadAttentionModel.to(device)
    loss_func_detection = torch.nn.CrossEntropyLoss()
    optim_task_detection = torch.optim.Adam(
      list(multiheadAttentionModel.parameters()), 
        lr=lr, weight_decay=l2
    )
    # ---  Model Training  ---

    loss_detection_total = 0
    best_acc = 0



    best_test_accuracy = 0.0
    best_model_state = None
    patience = 3  # 设置等待的轮数
    no_improvement_count = 0  # 记录验证集性能没有提升的轮数
    for epoch in range(num_epoch):
        multiheadAttentionModel.train()       
        corrects_pre_detection = 0
        loss_detection_total = 0
        detection_count = 0
        pre_label_detection= []
        labels= []
        c = 0
        for idx, batch in enumerate(train_loader):

            image = batch[0]
            text = batch[1]
            label = batch[2]
            #text  = text.to(device)
            #image = image.to(device)
            label = label.type(torch.LongTensor)
            label = label.to(device)

            text_features = torch.squeeze(text) #(64,1,512)-->(64,512)
            image_features = torch.squeeze(image)
            text_features = text_features.to(device)
            image_features = image_features.to(device)


            pre_detection = multiheadAttentionModel(text_features, image_features)

            loss_detection = loss_func_detection(pre_detection, label)

            optim_task_detection.zero_grad()
            loss_detection.backward()
            optim_task_detection.step()

            label = label.cpu().tolist()
            pre_detection = torch.argmax(pre_detection,dim=-1).cpu().tolist()


            train_loss = loss_detection.item()
            print("train iter: {}, train loss: {}".format(c+1, train_loss))
            loss_detection_total += train_loss

            labels.extend(label)
            pre_label_detection.extend(pre_detection)
            c += 1
        loss_detection_total /= c
        print("train_total_loss: ", loss_detection_total)
        train_report = classification_report(labels, pre_label_detection,output_dict=True,
                                                         target_names=["false","true"])
        print("epoch_end report: ", train_report)
        # ---  Test  ---
        loss_detection_test,  test_report = test(
            multiheadAttentionModel, test_loader)
        test_accuracy = test_report['accuracy']

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_model_state = multiheadAttentionModel.state_dict()
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        print("Epoch: {}, Test Accuracy: {:.4f}".format(epoch + 1, test_accuracy))

        # 如果验证集性能没有提升超过设定的等待轮数，停止训练
        if no_improvement_count >= patience:
            print("Early stopping! No improvement in validation performance for {} epochs.".format(patience))
            break

        # 保存最佳模型状态


    if best_model_state is not None:
        torch.save(best_model_state, "/media/qust521/qust_3_big/fake_news_data/weibo/nlpccf/weibo/best_model.pth")

        # ---  Output  ---


def test(multiheadAttentionModel, test_loader):
    # chineseclip_model2.eval()
    multiheadAttentionModel.eval()
    # classifier.eval()

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
    pre_label_detection = []
    labels = []
    c = 0



    with torch.no_grad():
        for  batch in test_loader:

            image = batch[0]
            text = batch[1]
            label = batch[2]

            label = label.type(torch.LongTensor)
            label = label.to(device)

            text_features = torch.squeeze(text) #(64,1,512)-->(64,512)
            image_features = torch.squeeze(image)
            text_features = text_features.to(device)
            image_features = image_features.to(device)

            # ---  TASK2 Detection  ---
            #text_features,image_features=chineseclip_model2(text,image)
            pre_detection = multiheadAttentionModel(text_features, image_features)

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
        loss_detection_test=loss_detection_total
        print("test_total_loss: ", loss_detection_total)
        test_report = classification_report(labels, pre_label_detection, output_dict=True,
                                             target_names=["false","true"])
        print("epoch_end report: ", test_report)




    return   loss_detection_test, test_report

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    train()
