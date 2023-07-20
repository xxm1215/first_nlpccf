import gc
import os
import warnings

import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from transformers import BertTokenizer, BertModel
import torchvision.models as models


warnings.filterwarnings('ignore')

# Configs

DEVICE = "cuda:0"
# DEVICE = "cpu"
NUM_WORKER = 1
BATCH_SIZE = 256
LR = 1e-3
L2 = 0  # 1e-5
NUM_EPOCH = 10
embed_dim = 512
num_heads = 1

class FakeDataset(Dataset):
    def __init__(self, df, folder_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.df = df
        self.folder_path = folder_path
        self.image_files = None
        self.image_ids = df['id'].tolist()
        self.max_size = 224

        # 图像预处理
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.model = BertModel.from_pretrained("bert-base-chinese")
        self.model = self.model.to(self.device)
        self.model.eval()

        self.resnet_model = models.resnet101(pretrained=True)
        self.resnet_model = self.resnet_model.to(self.device)
        self.resnet_model.eval()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]['text']
        label = self.df.iloc[idx]['label']
        image_id = self.df.iloc[idx]['id']

        text_input = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(self.device)

        if self.image_files is None:
            self.image_files = os.listdir(self.folder_path)
        filename = self.image_files[idx]
        image = Image.open(os.path.join(self.folder_path, filename)).convert("RGB")
        image = self.preprocess(image)
        image = image.unsqueeze(0).to(self.device)

        # 提取图像特征
        with torch.no_grad():
            features = self.resnet_model.conv1(image)
            features = self.resnet_model.bn1(features)
            features = self.resnet_model.relu(features)
            features = self.resnet_model.maxpool(features)

            features = self.resnet_model.layer1(features)
            features = self.resnet_model.layer2(features)
            features = self.resnet_model.layer3(features)
            image_features = self.resnet_model.layer4(features)

            text_features = self.model(**text_input)
            text_features = text_features.pooler_output  # extract pooled output
            text_features = text_features.unsqueeze(0)

        return image_features, text_features, label

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiheadAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.fc1 = nn.Linear(embed_dim * 2, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, text_feature, img_feature):
        text_feature = text_feature.unsqueeze(1)
        img_feature = img_feature.unsqueeze(1)#1,1,512

        text_attn_output, _ = self.multihead_attn(img_feature, text_feature, text_feature)
        img_attn_output, _ = self.multihead_attn(text_feature, img_feature, img_feature)

        text_attn_output = text_attn_output.squeeze(1)
        img_attn_output = img_attn_output.squeeze(1)

        features = torch.cat((img_attn_output, text_attn_output), dim=-1)

        x = torch.relu(self.fc1(features))
        x = self.fc2(x)
        return x

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
            pre_detection = model(text_features, image_features)
            loss_detection = loss_func_detection(pre_detection, label)

            optim_task_detection.zero_grad()
            loss_detection.backward()
            optim_task_detection.step()

            label = label.cpu().tolist()
            pre_detection = torch.argmax(pre_detection, dim=-1).cpu().tolist()

            train_loss = loss_detection.item()
            print("Train_Epoch: {}, train iter: {}, train loss: {}".format(epoch, c + 1, train_loss))
            loss_detection_total += train_loss
            labels.extend(label)
            pre_label_detection.extend(pre_detection)
            c += 1


        loss_detection_total /= c
        print("train_total_loss: ", loss_detection_total)
        train_report = classification_report(labels, pre_label_detection, output_dict=True,
                                             target_names=["true", "false"])
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


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    device = torch.device("cuda:0")

    # Configs

    DEVICE = "cuda:0"
    NUM_WORKER = 1
    BATCH_SIZE = 256
    LR = 1e-3
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


    folder_path = '/media/qust521/qust_3_big/fake_news_data/weibo/nlpccf/weibo17/images'



    df1 = pd.read_csv('/media/qust521/qust_3_big/fake_news_data/weibo/nlpccf/weibo17/tweets/train.csv',
                      encoding='GBK',encoding_errors='ignore')
    nan_rows1 = df1['text'].isnull()
    df1 = df1[~nan_rows1]


    df2 = pd.read_csv('/media/qust521/qust_3_big/fake_news_data/weibo/nlpccf/weibo17/tweets/test.csv',
                      encoding='GBK',encoding_errors='ignore')
    nan_rows2 = df2['text'].isnull()
    df2 = df2[~nan_rows2]

    train_set = FakeDataset(df1, folder_path)

    test_set = FakeDataset(df2, folder_path)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    test_loader = DataLoader(
        test_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    # ---  Build Model & Trainer  ---

    multiheadAttentionModel = MultiheadAttention(embed_dim, num_heads)
    multiheadAttentionModel.to(device)
    loss_func_detection = torch.nn.CrossEntropyLoss()
    optim_task_detection = torch.optim.Adam(
        list(multiheadAttentionModel.parameters()),
        lr=lr, weight_decay=l2
    )

    train(model=multiheadAttentionModel, train_loader=train_loader, test_loader=test_loader)
