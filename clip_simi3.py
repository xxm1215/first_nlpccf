import gc
import json
import re
import shutil
import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import clip
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.data import Dataset
import os
import pandas as pd
import torch
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from PIL import Image
from torchvision import transforms
import warnings


warnings.filterwarnings('ignore')



# 数据集
def clean_text(text):
    # 去除特殊字符和标点符号
    cleaned_text = re.sub(r"[^\u4e00-\u9fa5]", "", text)

    # cleaned_text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return cleaned_text

# def preprocess_text(text):
#     # 标记化
#     tokens = word_tokenize(text)
#
#     # 词干提取
#     stemmer = PorterStemmer()
#     stemmed_tokens = [stemmer.stem(token) for token in tokens]
#
#     # 连接词干化后的单词
#     processed_text = " ".join(stemmed_tokens)
#     return processed_text


# class FakeDataset(Dataset):
#     def __init__(self, df, folder_path):
#         self.device = torch.device("cuda:0")
#
#         # print(df)
#         self.folder_path = folder_path
#         self.image_files = None
#         # self.image_ids = df['id'].tolist()
#         self.max_size = 224
#         self.transform = transforms.Compose([
#             transforms.Resize((self.max_size, self.max_size)),  # 调整图像大小为最大尺寸
#             transforms.ToTensor()
#         ])
#         self.model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
#         self.preprocess = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
#         self.model = self.model.to(self.device)
#         self.model.eval()
#         with open(df,'r') as file:
#             json_data =json.load(file)
#             self.df = json_data
#
#         print("Found {} of documents.".format(len(self.df)))
#     def __len__(self):
#
#         return len(self.df)

    # def __getitem__(self, idx):
    #     item=self.df[idx]
    #     text =item['text']
    #     image_id =item['id']
    #     label=item['label']
    #
    #     #
    #     # image_id = self.df.iloc[idx]['id']
    #     # # print("image_id:", image_id)
    #     # text = self.df.iloc[idx]['text']
    #     # text = clean_text(text)
    #     # # print("text:",text)
    #     # label = self.df.iloc[idx]['label']
    #     # # print("label:",label)
    #     # # image_id = self.df.iloc[idx]['id']
    #     # #
    #     # print("image_id:",image_id)
    #     # filename=str(image_id)+'.jpg'
    #     filename ='{}.jpg'.format(image_id)
    #
    #     image = Image.open(os.path.join(self.folder_path, filename)).convert("RGB")
    #
    #     #
    #     # if self.image_files is None:
    #     #     self.image_files = os.listdir(self.folder_path)
    #     # filename = self.image_files[idx]
    #     # print("filename:",filename)
    #     # image = Image.open(os.path.join(self.folder_path, filename)).convert("RGB")
    #
    #     image = self.transform(image)
    #     # print("image shape:",image.shape)
    #
    #     image_input = self.preprocess(images=image, return_tensors="pt")
    #     image_input = image_input.to(self.device)
    #     image_features = self.model.get_image_features(**image_input)
    #     print(image_features.shape)
    #     # image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)  # normalize
    #
    #     text_input = self.preprocess(text=text, padding=True, return_tensors="pt")
    #     print(text_input)
    #     text_input = text_input.to(self.device)
    #     text_features = self.model.get_text_features(**text_input)
    #     print(text_features)
    #     # text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    #     text_features = text_features.detach()
    #     image_features = image_features.detach()
    #
    #     # image_features = image_features.to(self.device)
    #     # text_features = text_features.to(self.device)
    #
    #     return image_features, text_features, label
class FakeDataset(Dataset):
    def __init__(self, df, folder_path,max_text_length):
        self.device = torch.device("cuda:0")

        # print(df)
        self.folder_path = folder_path
        self.image_files = None
        self.max_size = 224
        self.transform = transforms.Compose([
            transforms.Resize((self.max_size, self.max_size)),  # 调整图像大小为最大尺寸
            transforms.ToTensor()
        ])
        self.max_text_length=max_text_length
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.model = self.model.to(self.device)
        self.model.eval()
        with open(df, 'r') as file:
            json_data = json.load(file)
            self.df = json_data

        print("Found {} of documents.".format(len(self.df)))

    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        item=self.df[idx]
        text =item['text']
        image_id =item['id']
        label=item['label']

        filename ='{}.jpg'.format(image_id)

        image = Image.open(os.path.join(self.folder_path, filename)).convert("RGB")

        image = self.preprocess(image).unsqueeze(0).to(device)
        # text = clip.tokenize(text).to(device)
        # 分割长文本序列
        # 假设模型的最大文本长度为 512
        if len(text) > self.max_text_length:
            text = text[:self.max_text_length]
        # text=self.preprocess(text)
        text = clip.tokenize(text).to(device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
        image_features = image_features.to(self.device)
        text_features = text_features.to(self.device)

        return image_features, text_features, label

class MLP(nn.Module):
    def __init__(self, embedding_dim, drop_out):
        super().__init__()

        self.fc = nn.Linear(embedding_dim *2, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, 2)
        self.drop = nn.Dropout(drop_out)

    def forward(self, text, img):

        features = torch.cat((text, img), dim=-1)# [batch_size, 1024]
        out = self.drop(self.fc(features))
        out = self.fc2(out)

        return out



class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiheadAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.fc1 = nn.Linear(embed_dim * 2, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, text_feature, img_feature):
        text_feature = text_feature.unsqueeze(1)
        img_feature = img_feature.unsqueeze(1)

        text_feature = text_feature.to(torch.float32)
        img_feature = img_feature.to(torch.float32)

        text_attn_output, _ = self.multihead_attn(img_feature, text_feature, text_feature)
        img_attn_output, _ = self.multihead_attn(text_feature, img_feature, img_feature)

        text_attn_output = text_attn_output.squeeze(1)
        img_attn_output = img_attn_output.squeeze(1)

        features = torch.cat((img_attn_output, text_attn_output), dim=-1)

        # sim_all = sim(text_attn_output, img_attn_output).to(device)  # torch.Size([256])
        #
        #
        # sim_mean = torch.mean(sim_all)
        # sim_std = torch.std(sim_all)
        # normalized_mini = sig((features - sim_mean) / sim_std)

        # output = normalized_mini * features
        output = features
        output = torch.relu(self.fc1(output))

        output = self.fc2(output)

        return output





def train(model, train_loader, test_loader):
    # ---  Load Config  ---


    # ---  Model Training  ---

    test_F1 = 0
    tolerant = 5
    saved_epoch = 0
    no_improvement_count = 0
    test_f1 = []
    best_test_accuracy = 0.0

    print("Start training..... ")
    for epoch in range(num_epoch):
        model.train()
        loss_detection_total = 0
        pre_label_detection = []
        labels = []
        c = 0

        for idx, batch in enumerate(train_loader):
            image = batch[0]
            text = batch[1]
            label = batch[2]


            label = torch.LongTensor(label)
            label = label.to(device)

            # train_loader.
            text_features = torch.squeeze(text)  # (64,1,512)-->(64,512)
            image_features = torch.squeeze(image)

            text_features = text_features.to(device)
            image_features = image_features.to(device)
            text_features = text_features.to(torch.float32)
            image_features =image_features.to(torch.float32)

            pre_detection = model(text_features, image_features)
            loss_detection = loss_func_detection(pre_detection, label)
            # print(loss_detection)
            optim_task_detection.zero_grad()
            loss_detection.backward()
            optim_task_detection.step()

            label = label.to(device).tolist()
            pre_detection = torch.argmax(pre_detection, dim=-1).to(device).tolist()

            # label = label.cpu().tolist()
            # pre_detection = torch.argmax(pre_detection, dim=-1).cpu().tolist()

            train_loss = loss_detection.item()
            print("Train_Epoch: {}, train iter: {}, train loss: {}".format(epoch, c + 1, train_loss))
            loss_detection_total += train_loss
            labels.extend(label)
            pre_label_detection.extend(pre_detection)
            c += 1


        loss_detection_total /= c
        print("train_total_loss: ", loss_detection_total)
        print("labels:",labels)
        print("pre_label_detection:",pre_label_detection)
        train_report = classification_report(labels, pre_label_detection, output_dict=True,
                                             target_names=["true", "false"])
        # train_report = classification_report(labels, pre_label_detection, labels=[0, 1])
        print("train_report:", train_report)

        # ---  Test  ---

        loss_detection_test, test_report = test(
        model, test_loader)
        print(test_report)
        test_accuracy = test_report['accuracy']
        # break
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_model_state = model.state_dict()
            no_improvement_count = 0
        else:
            no_improvement_count += 1


        print("Epoch: {}, Test Accuracy: {:.4f}".format(epoch + 1, test_accuracy))

        performance_metrics_1 = test_report["false"]
        df3 = pd.DataFrame(performance_metrics_1,index=['false'])
        df3.to_csv("/media/qust521/qust_3_big/fake_news_data/weibo/nlpccf/weibo/data_simi/test_false.csv", index=False)
        performance_metrics_2 = test_report["true"]
        df4 = pd.DataFrame(performance_metrics_2,index=['true'])
        df4.to_csv("/media/qust521/qust_3_big/fake_news_data/weibo/nlpccf/weibo/data_simi/test_true.csv", index=False)

        f1_macro = test_report['macro avg']['f1-score']
        print("macro F1 is: ", f1_macro)
        test_f1.append(f1_macro)
        if f1_macro > test_F1:
            test_F1 = f1_macro
            saved_model = model
            saved_epoch = epoch
            print("saving model with test_f1 {} at Epoch {}".format(test_F1, saved_epoch + 1))
            torch.save(saved_model, "/media/qust521/qust_3_big/fake_news_data/weibo/nlpccf/weibo/data_simi/best_model.pth" )


        if epoch-saved_epoch >= tolerant:
            model.cpu()
            saved_model.cpu()


            del model, saved_model,pre_detection, loss_detection, labels, train_loader, test_loader, loss_detection_total
            gc.collect()
            torch.cuda.empty_cache()
            max_F1 =max(test_f1)
            with open("/media/qust521/qust_3_big/fake_news_data/weibo/nlpccf/weibo/data_simi/test_finetune_t1.txt", 'w') as outf:
                outf.write("Best F1 score: {} at epoch {}\n".format(max_F1, epoch + 1))

            print("Early stopping at epoch {}.".format(epoch + 1))
            break

        # ---  Output  ---


def test(model, test_loader):
    print("Start to testing.... ")
    model.eval()
    loss_detection_total = 0
    pre_label_detection = []
    labels = []
    c = 0


    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            image = batch[0]
            text = batch[1]
            label = batch[2]

            label = label.type(torch.LongTensor)
            label = label.to(device)

            text_features = torch.squeeze(text)  # (64,1,512)-->(64,512)
            image_features = torch.squeeze(image)
            text_features = text_features.to(device)
            image_features = image_features.to(device)
            text_features = text_features.to(torch.float32)
            image_features = image_features.to(torch.float32)
            # ---  TASK2 Detection  ---
            pre_detection = model(text_features, image_features)

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
                                            target_names=["true", "false"])

    return loss_detection_test, test_report

def calculate_class_weights(train_loader):
    num_classes=2
    class_counts = torch.zeros(num_classes)  # Initialize class counts with

    total_samples = 0

    for batch in train_loader:
        label = batch[2]
        class_counts += torch.bincount(label, minlength=num_classes)
        total_samples += label.size(0)

    class_weights = total_samples / (num_classes * class_counts)

    return class_weights


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    device = torch.device("cuda:0")

    # Configs

    DEVICE = "cuda:0"
    NUM_WORKER = 1
    BATCH_SIZE = 8
    LR = 1e-5
    L2 = 0  # 1e-5
    NUM_EPOCH = 30
    embed_dim = 512
    num_heads = 1

    sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    sig = torch.nn.Sigmoid()
    num_workers = NUM_WORKER
    batch_size = BATCH_SIZE
    lr = LR
    l2 = L2
    num_epoch = NUM_EPOCH


    # ---  Load Data  ---


    folder_path = '/media/qust521/qust_3_big/fake_news_data/weibo/nlpccf/poli/poli_img_all'

    df1 = '/media/qust521/qust_3_big/fake_news_data/weibo/nlpccf/poli/train.json'
    df2 = '/media/qust521/qust_3_big/fake_news_data/weibo/nlpccf/poli/test.json'
    #
    # df1 = pd.read_csv('/media/qust521/qust_3_big/fake_news_data/weibo/nlpccf/poli/train.json',
    #                   encoding='utf-8',encoding_errors='ignore')
    # nan_rows1 = df1['text'].isnull()
    # df1 = df1[~nan_rows1]
    #
    #
    # df2 = pd.read_csv('/media/qust521/qust_3_big/fake_news_data/weibo/nlpccf/poli/test.json',
    #                   encoding='utf-8',encoding_errors='ignore')
    # nan_rows2 = df2['text'].isnull()
    # df2 = df2[~nan_rows2]

    train_set = FakeDataset(df1, folder_path, max_text_length = 77 )

    test_set = FakeDataset(df2, folder_path, max_text_length = 77 )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=True)
    # ---  Build Model & Trainer  ---

    N_FILTERS = 100
    FILTER_SIZES = [2, 3, 4]
    OUTPUT_DIM = 2
    DROPOUT = 0.2
    EMBEDDING_DIM = 512
    model_cnn = MLP(EMBEDDING_DIM, DROPOUT)
    model_cnn.to(device)

    multiheadAttentionModel = MultiheadAttention(embed_dim, num_heads)
    multiheadAttentionModel.to(device)
    # class_weights=calculate_class_weights(train_loader)
    # loss_func_detection = torch.nn.CrossEntropyLoss(weight=class_weights)
    loss_func_detection = torch.nn.CrossEntropyLoss()
    optim_task_detection = torch.optim.Adam(
        list(multiheadAttentionModel.parameters()),
        lr=lr, weight_decay=l2
    )

    train(model=model_cnn, train_loader=train_loader, test_loader=test_loader)
