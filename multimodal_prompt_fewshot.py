import glob, json, random, re, nltk, csv, argparse
import numpy as np
from openprompt.data_utils import InputExample, FewShotSampler
from openprompt.prompts import MixedTemplate, SoftTemplate, ManualTemplate
from openprompt.plms import load_plm
from openprompt import PromptDataLoader, PromptForClassification
from openprompt.prompts import SoftVerbalizer, ManualVerbalizer, KnowledgeableVerbalizer
import torch
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import clip
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"
## fakeddit data scripts
    # image_files = glob.glob("./images_500/images/images/*.jpg")
    # all_data = []
    # with open("image_info.csv", 'r') as inf:
    #     data = csv.reader(inf)
    #     next(data)
    #     for line in data:
    #         text = line[6]
    #         image_id = line[10]
    #         label = line[18] # 0-true, 1-fake
    #         d = {}
    #
    #         if len(text) > 10 and image_id + ".jpg" in [name.split("/")[-1] for name in image_files]:
    #             d["id"] = image_id
    #             d["txt"] = text
    #             d['label'] = label
    #         else:
    #             continue
    #         all_data.append(d)

def xrange(x):
    return iter(range(x))

def test_set(all, train, dev, few_shot=True, if_dev=True):
    test = []
    if few_shot:
        used_id = []
        for idx in range(len(train)):
            used_id.append(train[idx].guid)
        if if_dev:
            for idx_1 in range(len(dev)):
                used_id.append(dev[idx_1].guid)

        for idx in range(len(all)):
            if all[idx].guid not in used_id:
                test.append(all[idx])
            else:
                continue
    return test

class Alpha(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_beta = torch.nn.Parameter(data=torch.Tensor(0), requires_grad=True)

    def forward(self):  # no inputs
        beta = torch.sigmoid(self.raw_beta)  # get (0,1) value
        return beta


class Proj_layers(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj1 = torch.nn.Linear(1024, 768, device=device)
        self.ln1 = torch.nn.LayerNorm(768, device=device)
        self.proj2 = torch.nn.Linear(768, 768, device=device)
        self.ln2 = torch.nn.LayerNorm(768, device=device)
        # self.sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        # self.sig = torch.nn.Sigmoid()

    def forward(self, txt, img):
        out_emb = torch.cat((img, txt), 1)
        # out_emb = img * txt
        # print(txt)
        # print(img)
        # print(out_emb.size())
        out_emb = self.ln1(F.relu(self.proj1(out_emb.float())))
        out_emb = self.ln2(F.relu(self.proj2(out_emb)))
        return out_emb

def mini_batching(inputs):
    mini_batch = []
    sim_all = []
    proj = Proj_layers()
    sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    sig = torch.nn.Sigmoid()
    for item in inputs['guid']:
        for sample in all_data:
            if item == sample['id']:
                i_input = preprocess(Image.open(image_path + item + ".jpg")).unsqueeze(0).to(device)
                t_input = clip.tokenize(sample['txt'], truncate=True).to(device)

                i_emb = model.encode_image(i_input)
                t_emb = model.encode_text(t_input)
                # out_emb = t_emb + i_emb
                # proj = torch.nn.Linear(512, 768, device=device)
                out_emb = proj(t_emb, i_emb)

                sim_all.append(sim(t_emb, i_emb))

                # out_emb = torch.cat((t_emb, i_emb),-1)
                mini_batch.append(out_emb)
    sim_all = torch.stack(sim_all).to(device).squeeze()
    mini_batch = torch.stack(mini_batch).to(device).squeeze()
    sim_mean = torch.mean(sim_all)
    sim_std = torch.std(sim_all)
    normalized_mini = sig((mini_batch - sim_mean) / sim_std)
    mini_batch = normalized_mini * mini_batch

    return mini_batch

def start_training(model, train_dataloader, val_dataloader,
          test_dataloader, loss_function, optimizer, alpha, epoch):

    print("alpha is {}".format(alpha))
    saved_model = None
    val_f1_macro_in_alpha = 0
    val_loss_in_alpha = 10
    tolerant = 5
    saved_epoch = 0
    for epoch in range(epoch):
        tot_loss = 0
        print("===========EPOCH:{}=============".format(epoch+1))
        for step, inputs in enumerate(train_dataloader):
            if use_cuda:
                inputs = inputs.cuda()

            out = model.forward_without_verbalize(inputs)
            mini_batch = mini_batching(inputs)
            out = alpha * mini_batch + out
            logits, draw = model.verbalizer.process_outputs(outputs=out, batch=inputs)

            # logits = prompt_model(inputs)
            labels = inputs['label']
            loss = loss_function(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            # if step % 50 == 0:
            #     print(tot_loss/(step+1))

        model.eval()

        allpreds = []
        alllabels = []
        eval_total_loss = 0
        dev_total_loss = 0
        c = 0
        for step, inputs in enumerate(val_dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            out = model.forward_without_verbalize(inputs) # [batch_size, seq_len, feature]

            mini_batch = mini_batching(inputs)
            out = alpha * out + mini_batch

            logits, draw = model.verbalizer.process_outputs(outputs=out, batch=inputs)
            labels = inputs['label']
            eval_loss = loss_function(logits, labels)
            eval_total_loss += eval_loss.item()
            alllabels.extend(labels.cpu().tolist())
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            dev_loss = eval_total_loss/(step+1
                                        )
            dev_total_loss += dev_loss
            c += 1

        dev_total_loss = dev_total_loss/(c+1)
        print("===========val_loss===========: ", dev_total_loss)
        acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
        print("validation:",  acc)
        report_val = classification_report(alllabels, allpreds, output_dict=True,
                                           labels=[0,1], target_names=["real", "fake"],
                                           )
        f1_fake = report_val['fake']['f1-score']
        f1_real = report_val['real']['f1-score']
        f1_macro = report_val['macro avg']['f1-score']
        if float(val_loss_in_alpha) > dev_total_loss:
            # val_f1_macro_in_alpha = float(f1_macro)
            val_loss_in_alpha = dev_total_loss
            saved_model = model
            saved_epoch = epoch
            print("saving model at {} alpha with {} val_loss at Epoch {}".format(alpha, val_loss_in_alpha, saved_epoch+1))
        if epoch - saved_epoch >= tolerant:
            print("Early stopping at epoch {}.".format(epoch+1))
            break

    allpreds = []
    alllabels = []
    alllogits = []
    for step, inputs in enumerate(test_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        out = saved_model.forward_without_verbalize(inputs)
        mini_batch = mini_batching(inputs)
        out = alpha * out + mini_batch
        # alllogits.extend(out.detach().cpu().numpy())

        logits, draw = saved_model.verbalizer.process_outputs(outputs=out, batch=inputs)
        alllogits.extend(draw.detach().cpu().numpy())
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())


    acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
    print("test:", acc)
    report_test = classification_report(alllabels, allpreds, labels=[0,1], target_names=["real", "fake"])

    if str(args.data) == "goss":
        if t_or_f(args.full):
            with open("./results/goss/full/{}_{}sd_{}al_{}.txt".format(args.data, args.seed,
                                                        args.alpha, args.full), "w") as out_file:
                print(report_test)
                out_file.write(report_test)
        else:
            with open("./results/goss/few/{}_{}st_{}sd_{}al_{}.txt".format(args.data, args.shot+args.shot, args.seed,
                                                        args.alpha, args.full), "w") as out_file:
                print(report_test)
                out_file.write(report_test)
    else:
        if t_or_f(args.full):
            with open("./results/poli/full/{}_{}sd_{}al_{}.txt".format(args.data, args.seed,
                                                        args.alpha, args.full), "w") as out_file:
                print(report_test)
                out_file.write(report_test)
        else:
            with open("./results/poli/few/{}_{}st_{}sd_{}al_{}.txt".format(args.data, args.shot+args.shot, args.seed,
                                                        args.alpha, args.full), "w") as out_file:
                print(report_test)
                out_file.write(report_test)

    return alllogits, allpreds


def tsne_plot(inputs, labels):
    "Creates and TSNE model and plots it"


    tsne_model = TSNE(perplexity=10, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(inputs)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(labels)):
        if labels[i] == 0:
            plt.scatter(x[i], y[i], c='b')
        elif labels[i] == 1:
            plt.scatter(x[i], y[i], c='r')
        # plt.annotate(labels[i],
        #              xy=(x[i], y[i]),
        #              xytext=(5, 2),
        #              textcoords='offset points',
        #              ha='right',
        #              va='bottom')
    plt.show()



def t_or_f(arg):
    ua = str(arg).upper()
    if 'TRUE'.startswith(ua):
       return True
    elif 'FALSE'.startswith(ua):
       return False

def get_data(path3):
    all_data = []
    with open(path3, 'r') as inf:
        data = csv.reader(inf)
        next(data)
        for line in data:
            text = line[1]
            image_id = line[2]
            label = line[3]  # 0-true, 1-fake
            d = {}

            if len(text) > 0 and image_id + ".jpg" in [name.split("/")[-1] for name in image_files]:
                d["id"] = image_id
                d["txt"] = text
                d['label'] = int(label)
            else:
                continue
            all_data.append(d)

    return all_data


def get_n_trainable_params(model):
    # all trainable
    num_total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # split into the plm and classisifcation head
    num_plm_trainable = sum(p.numel() for p in model.plm.parameters() if p.requires_grad)

    # template trainable
    try:
        num_template_trainable = sum(p.numel() for p in model.template.soft_embedding.parameters() if p.requires_grad)
    except:
        num_template_trainable = 0

    # verbalizer trainable
    num_verbalizer_trainable = sum(p.numel() for p in model.verbalizer.parameters() if p.requires_grad)

    # assert sum of the two = total
    assert num_plm_trainable + num_template_trainable + num_verbalizer_trainable == num_total_trainable

    print(f"Number of trainable parameters of PLM: {num_plm_trainable}\n")
    print('#' * 50)
    print(f"Number of trainable parameters of template: {num_template_trainable}\n")
    print('#' * 50)
    print(f"Number of trainable parameters of verbalizer: {num_verbalizer_trainable}\n")
    print('#' * 50)
    print(f"Total number of trainable parameters of whole model: {num_total_trainable}")
    print(f"Verbalizer grouped_parameters_1 require_grad: {model.verbalizer.group_parameters_1[0].requires_grad}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="help")
    parser.add_argument("--alpha", type=float, help="alpha from 0-1")
    parser.add_argument("--seed",  type=int, help="seed from 1-5")
    parser.add_argument("--shot", type=int, help="shot from 1, 2, 4, 8, 50")
    parser.add_argument("--full", type=str, default=False)
    parser.add_argument("--data", type=str, default="goss")
    args = parser.parse_args()

    print("<<<<<<<< Data:{}, Shot:{}, Seed:{}, Alpha:{}, Full:{}".format(args.data, args.shot+args.shot, args.seed,
                                                                         args.alpha, args.full))

    ## fakenewsnet data scripts

    goss_path1 = "../FakeNewsNet-master/code/fakenewsnet_dataset/gossipcop_multi/goss_img_all/*.jpg"
    goss_path2 = "../FakeNewsNet-master/code/fakenewsnet_dataset/gossipcop_multi/goss_img_all/"
    goss_path3 = '../FakeNewsNet-master/gossipcop_multi.csv'

    poli_path1 = "../FakeNewsNet-master/code/fakenewsnet_dataset/politifact_multi/poli_img_all/*.jpg"
    poli_path2 = "../FakeNewsNet-master/code/fakenewsnet_dataset/politifact_multi/poli_img_all/"
    poli_path3 = '../FakeNewsNet-master/politifact_multi.csv'

    if str(args.data) == "goss":
        image_files = glob.glob(goss_path1)
        image_path = goss_path2
        all_data = get_data(goss_path3)
    else:
        image_files = glob.glob(poli_path1)
        image_path = poli_path2
        all_data = get_data(poli_path3)


    model, preprocess = clip.load("ViT-B/32", device=device)

    dataset = []
    for idx, d in enumerate(all_data):
        input_example = InputExample(text_a=d['txt'], label=int(d['label']), guid=d['id'])
        dataset.append(input_example)

    if t_or_f(args.full):
        train, dev = train_test_split(dataset, test_size=0.2, shuffle=True)
        dev, test = train_test_split(dev, test_size=0.5, shuffle=True)

    else:
        sampler = FewShotSampler(num_examples_per_label=args.shot, num_examples_per_label_dev=args.shot, also_sample_dev=True)
        train, dev = sampler.__call__(train_dataset=dataset, seed=args.seed)

        test = test_set(dataset, train, dev, if_dev=True)

    pre_lm_dir = "roberta-base"

    plm, tokenizer, model_config, WrapperClass = load_plm("roberta", pre_lm_dir)
    # mytemplate = ManualTemplate(tokenizer=tokenizer, text='Here is a piece of news with {"mask"} information.{"placeholder":"text_a"}')
    mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer,
                               # text='{"soft":"<head>"}Here is a piece of news with {"mask"} information.{"soft":"<tail>"} {"placeholder":"text_a"} {"placeholder":"text_b"}')
                               # text='{"soft"}{"soft"}{"soft"}{"soft"}{"soft"}{"soft"}{"soft"}{"soft"}{"soft"}{"soft"}{"soft"}{"soft"}{"soft"}{"soft"}{"soft"}{"soft"}{"soft"}{"soft"}{"soft"}{"soft"}{"mask"}{"placeholder":"text_a"}')
                               text='{"soft": None, "duplicate": 20}{"mask"}{"placeholder":"text_a"}')
    #                            text='Here is a piece of news with {"mask"} information.{"placeholder":"text_a"}')
    #                            text='{"soft": None, "duplicate": 20, "post_processing": "lstm"} {"mask"}. {"placeholder":"text_a"}')
    # mytemplate = SoftTemplate(model=plm, tokenizer=tokenizer, text='{"soft"}{"soft"}{"soft"}{"soft"}{"mask"}{"placeholder":"text_a"}')

    train_dataloader = PromptDataLoader(dataset=train, template=mytemplate, tokenizer=tokenizer,
                                        tokenizer_wrapper_class=WrapperClass, max_seq_length=512, decoder_max_length=3,
                                        batch_size=4, shuffle=True, teacher_forcing=False, predict_eos_token=False,
                                        truncate_method="tail")

    validation_dataloader = PromptDataLoader(dataset=dev, template=mytemplate, tokenizer=tokenizer,
                                             tokenizer_wrapper_class=WrapperClass, max_seq_length=512,
                                             decoder_max_length=3,
                                             batch_size=4, shuffle=True, teacher_forcing=False, predict_eos_token=False,
                                             truncate_method="tail")

    test_dataloader = PromptDataLoader(dataset=test, template=mytemplate, tokenizer=tokenizer,
                                       tokenizer_wrapper_class=WrapperClass, max_seq_length=512, decoder_max_length=3,
                                       batch_size=4, shuffle=True, teacher_forcing=False, predict_eos_token=False,
                                       truncate_method="tail")

    myverbalizer = SoftVerbalizer(tokenizer, plm, num_classes=2)
    # myverbalizer = ManualVerbalizer(tokenizer=tokenizer, classes=["real", "fake"],
    #                                 label_words= {
    #                                     "real": ["actual", "true", "genuine", "really", "tangible", "realistic", "reality", "veridical", "material" ,"very"],
    #                                     "fake": ["counterfeit", "sham", "phony", "false", "bogus", "cheat", "imposter", "imitation", "impostor", "falsify"]
    #                                 })
    # myverbalizer = KnowledgeableVerbalizer(tokenizer, num_classes=2).from_file("./knowlegeable_verbalizer.txt")

    use_cuda = True
    prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=True)
    if use_cuda:
        prompt_model = prompt_model.cuda()

    loss_func = torch.nn.CrossEntropyLoss()
    no_decay = ['bias', 'LayerNorm.weight']
    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    # Using different optimizer for prompt parameters and model parameters
    # optimizer_grouped_parameters2 = [
    #     {'params': prompt_model.verbalizer.group_parameters_1, "lr": 3e-5},
        # {'params': prompt_model.verbalizer.group_parameters_2, "lr": 3e-4},
    # ]
    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5)

    alpha = args.alpha
    get_n_trainable_params(prompt_model)

    all_logits, all_preds = start_training(model=prompt_model, train_dataloader=train_dataloader, val_dataloader=validation_dataloader,
          test_dataloader=test_dataloader, loss_function=loss_func, optimizer=optimizer1, alpha=alpha, epoch=20)

    get_n_trainable_params(prompt_model)
    # tsne_plot(np.array(all_logits), all_preds)
