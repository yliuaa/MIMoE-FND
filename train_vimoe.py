import os
import pickle as pickle
import random
import time
import argparse
import datetime

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.autograd import Variable
from torch.utils.data import DataLoader
from transformers import BertTokenizer, ChineseCLIPImageProcessor, CLIPProcessor

from util import Progbar
import pytorch_warmup as warmup
from models.vimoe_v2 import Vimoe_V2

# constants
stateful_metrics = [
    "L-RealTime",
    "lr",
    "APEXGT",
    "empty",
    "exclusion",
    "FW1",
    "QF",
    "QFGT",
    "QFR",
    "BK1",
    "FW",
    "BK",
    "FW1",
    "BK1",
    "LC",
    "Kind",
    "FAB1",
    "BAB1",
    "A",
    "AGT",
    "1",
    "2",
    "3",
    "4",
    "0",
    "gt",
    "pred",
    "RATE",
    "SSBK",
]
GT_size = 224
word_token_length = 197
image_token_length = 197
token_chinese = BertTokenizer.from_pretrained("bert-base-chinese")
token_uncased = BertTokenizer.from_pretrained("bert-base-uncased")
clip_chinese_processor = ChineseCLIPImageProcessor(do_rescale=False, do_resize=False)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")


# Helper functions
def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def to_np(x):
    return x.data.cpu().numpy()

def collate_fn_english(data):
    sents = [i[0][0] for i in data]
    image = [i[0][1] for i in data]
    image_aug = [i[0][2] for i in data]
    labels = [i[0][3] for i in data]
    category = [0 for i in data]
    GT_path = [i[1] for i in data]
    token_data = token_uncased.batch_encode_plus(
        batch_text_or_text_pairs=sents,
        truncation=True,
        padding="max_length",
        max_length=word_token_length,
        return_tensors="pt",
        return_length=True,
    )
    clip_inputs = clip_processor(
        text=sents,
        images=image,
        truncation=True,
        padding="max_length",
        max_length=77,
        return_tensors="pt",
        return_length=True,
    )

    input_ids = token_data["input_ids"]
    attention_mask = token_data["attention_mask"]
    token_type_ids = token_data["token_type_ids"]
    image = torch.stack(image)
    image_aug = torch.stack(image_aug)
    labels = torch.LongTensor(labels)
    category = torch.LongTensor(category)

    return (
        (input_ids, attention_mask, token_type_ids),
        (image, image_aug, labels, category, sents),
        clip_inputs,
        GT_path,
    )

def collate_fn_chinese(data):
    sents = [i[0][0] for i in data]
    image = [i[0][1] for i in data]
    image_aug = [i[0][2] for i in data]
    labels = [i[0][3] for i in data]
    category = [0 for i in data]
    GT_path = [i[1] for i in data]
    token_data = token_chinese.batch_encode_plus(
        batch_text_or_text_pairs=sents,
        truncation=True,
        padding="max_length",
        max_length=word_token_length,
        return_tensors="pt",
        return_length=True,
    )
    clip_img_inputs = clip_chinese_processor.preprocess(
        images=image, return_tensors="pt"
    )

    input_ids = token_data["input_ids"]
    attention_mask = token_data["attention_mask"]
    token_type_ids = token_data["token_type_ids"]

    image = torch.stack(image)
    image_aug = torch.stack(image_aug)
    labels = torch.LongTensor(labels)
    category = torch.LongTensor(category)

    return (
        (input_ids, attention_mask, token_type_ids),
        (image, image_aug, labels, category, sents),
        clip_img_inputs,
        GT_path,
    )

def load_model(model, load_path, strict=False):
    load_net = torch.load(load_path)
    model.load_state_dict(load_net, strict=strict)

def main(args):
    print(args)
    # ====================== Reproducibility ======================
    seed = 2024
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    setting = {}
    setting["checkpoint_path"] = args.checkpoint
    if len(setting["checkpoint_path"]) != 0:
        print("loading checkpoint from {}".format(setting["checkpoint_path"]))
    setting["train_dataname"] = args.train_dataset
    setting["val_dataname"] = args.test_dataset
    setting["val"] = args.val
    setting["duplicate_fake_times"] = args.duplicate_fake_times
    setting["data_augment"] = False
    setting["is_use_bce"] = True
    ######## ADDITIONAL FEATURES ###########
    setting["get_MLP_score"] = args.get_MLP_score
    setting["device"] = args.device
    custom_batch_size = args.batch_size
    custom_num_epochs = args.epochs

    # ====================== Data Preparation ======================
    train_dataset, validate_dataset, train_loader, validate_loader = (
        None,
        None,
        None,
        None,
    )
    shuffle, num_workers = True, 4
    train_sampler = None

    # training dataset
    if setting["train_dataname"][:5] == "weibo":
        print("Using weibo as training")
        # Note: bert-base-chinese is within MixSet_dataset
        from data.weibo_dataset import weibo_dataset

        train_dataset = weibo_dataset(
            is_train=True,
            image_size=GT_size,
            dataset=setting["train_dataname"],
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=custom_batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn_chinese,
            num_workers=num_workers,
            sampler=train_sampler,
            drop_last=True,
            pin_memory=True,
        )
        setting["thresh"] = 0.5
        print(f"thresh:{setting['thresh']}")
    else:
        print("Using GossipCop as training")
        from data.FakeNet_dataset import FakeNet_dataset
        train_dataset = FakeNet_dataset(
            is_train=True,
            dataset=setting["train_dataname"],
            image_size=GT_size,
            data_augment=setting["data_augment"],
            duplicate_fake_times=setting["duplicate_fake_times"],
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=custom_batch_size,
            shuffle=True,
            collate_fn=collate_fn_english,
            num_workers=4,
            sampler=None,
            drop_last=True,
            pin_memory=True,
        )
        setting["thresh"] = train_dataset.thresh
        print(f"thresh:{setting['thresh']}")

    # VALIDATION DATASET
    if setting["val_dataname"][:5] == "weibo":
        print("Using weibo as validation")
        from data.weibo_dataset import weibo_dataset

        validate_dataset = weibo_dataset(
            is_train=False,
            image_size=GT_size,
            dataset=setting["train_dataname"],
        )
        print(len(validate_dataset))
        validate_loader = DataLoader(
            validate_dataset,
            batch_size=custom_batch_size,
            shuffle=False,
            collate_fn=collate_fn_chinese,
            num_workers=4,
            sampler=None,
            drop_last=False,
            pin_memory=True,
        )

    else:
        from data.FakeNet_dataset import FakeNet_dataset
        print("using GossipCop as validation")
        validate_dataset = FakeNet_dataset(
            is_train=False,
            dataset=setting["val_dataname"],
            image_size=GT_size,
        )
        validate_loader = DataLoader(
            validate_dataset,
            batch_size=custom_batch_size,
            shuffle=False,
            collate_fn=collate_fn_english,
            num_workers=4,
            sampler=None,
            drop_last=False,
            pin_memory=True,
        )


    # ====================== Model ======================
    print("building ViMoE V2 model")
    model = Vimoe_V2(
        dataset=setting["train_dataname"],
        text_token_len=word_token_length,
        image_token_len=image_token_length,
        is_use_bce=setting["is_use_bce"],
        batch_size=custom_batch_size,
        thresh=setting["thresh"],
        agr_threshold=args.agr_threshold,
        sem_threshold=args.sem_threshold,
        warmup_epochs=0
    )

    if len(setting["checkpoint_path"]) != 0:
        print("loading checkpoint: {}".format(setting["checkpoint_path"]))
        load_model(model, setting["checkpoint_path"])
    model = model.cuda()
    model.train()


    # ====================== Loss and Optimizer ======================
    criterion = nn.BCEWithLogitsLoss(train_dataset.pos_weight).cuda()
    optim_params_normal, optim_params_fast, optim_params_extremefast = [], [], []
    name_params_normal, name_params_fast, name_params_extremefast = [], [], []

    finetune_encoders = False
    for k, v in model.named_parameters():
        if v.requires_grad:
            if "image_model" in k or "text_model" in k:
                finetune_encoders = True
                name_params_normal.append(k)
                optim_params_normal.append(v)
            elif "interaction" in k:
                name_params_extremefast.append(k)
                optim_params_extremefast.append(v)
            else:
                name_params_fast.append(k)
                optim_params_fast.append(v)
    fine_tuning = args.finetune > 0
    print(f"THE CURRENT MODE FOR FINETUNING:{fine_tuning}")
    num_steps = int(len(train_loader) * custom_num_epochs * 1.1)
    if finetune_encoders:
        optimizer = torch.optim.AdamW(
            optim_params_normal, lr=1e-5, betas=(0.9, 0.999), weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
        warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
    optimizer_fast = torch.optim.AdamW(
        optim_params_fast,
        lr=5e-5 if not fine_tuning > 0 else 1e-5,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )
    optimizer_extremefast = torch.optim.AdamW(
        optim_params_extremefast,
        lr=args.int_lr if not fine_tuning else 1e-5,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )
    scheduler_fast = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_fast, T_max=num_steps
    )
    warmup_scheduler_fast = warmup.UntunedLinearWarmup(optimizer_fast)
    scheduler_extremefast = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_extremefast, T_max=num_steps
    )
    warmup_scheduler_extremefast = warmup.UntunedLinearWarmup(optimizer_extremefast)


    # ====================== Training ======================
    print("loader size " + str(len(train_loader)))
    best_validate_acc = 0.000
    best_acc_so_far = 0.000
    best_img_acc = 0.000
    best_text_acc = 0.000
    best_epoch_record = 0
    global_step = 0
    print("training model")

    if setting["val"] != 0:
        custom_num_epochs = 1

    val_losses = []
    for epoch in range(custom_num_epochs):
        cost_vector = []
        acc_vector = []
        int_beta = args.int_beta
        if setting["val"] == 0:
            total = len(train_dataset)
            progbar = Progbar(total, width=10, stateful_metrics=stateful_metrics)
            times = []
            for i, items in enumerate(train_loader):

                with torch.enable_grad():
                    logs = []
                    model.eval()
                    """
                    (input_ids, attention_mask, token_type_ids), (image, labels, category, sents)
                    """
                    texts, others, clip_inputs, GT_path = items
                    input_ids, attention_mask, token_type_ids = texts
                    image, image_aug, labels, category, sents = others
                    (
                        input_ids,
                        attention_mask,
                        token_type_ids,
                        image,
                        image_aug,
                        labels,
                        category,
                        clip_inputs,
                    ) = (
                        to_var(input_ids),
                        to_var(attention_mask),
                        to_var(token_type_ids),
                        to_var(image),
                        to_var(image_aug),
                        to_var(labels),
                        to_var(category),
                        clip_inputs.to(args.device),
                    )
                    
                    (
                        mix_output,
                        image_only_output,
                        text_only_output,
                        loss_int,
                    ) = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        image=image,
                        clip_inputs=clip_inputs,
                    )

                    if setting["is_use_bce"]:
                        labels = labels.float().unsqueeze(1)

                    # =================== Loss Calculation ===================
                    loss_CE = criterion(mix_output, labels)
                    loss_CE_image = criterion(image_only_output, labels)
                    loss_CE_text = criterion(text_only_output, labels)
                    loss_single_modal = (loss_CE_text + loss_CE_image) / 2
                    loss = loss_CE + 1.0 * loss_single_modal + int_beta * loss_int

                    # =================== Backward ===================
                    global_step += 1
                    if finetune_encoders:
                        optimizer.zero_grad()
                    optimizer_fast.zero_grad()
                    optimizer_extremefast.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                    if epoch >= 10 and finetune_encoders:
                        # fine-tune MAE and BERT from the 5th epoch
                        optimizer.step()
                    optimizer_fast.step()
                    optimizer_extremefast.step()

                    logs.append(("CE_loss", loss_CE.item()))
                    logs.append(("Image", loss_CE_image.item()))
                    logs.append(("Text", loss_CE_text.item()))
                    logs.append(("Int_loss", int_beta * loss_int.item()))
                    if not setting["is_use_bce"]:
                        _, argmax = torch.max(mix_output, 1)
                        accuracy = (labels == argmax.squeeze()).float().mean()
                    else:
                        accuracy = (
                            (torch.sigmoid(mix_output).round_() == labels.round_())
                            .float()
                            .mean()
                        )

                    cost_vector.append(loss.item())
                    acc_vector.append(accuracy.item())
                    mean_cost, mean_acc = np.mean(cost_vector), np.mean(acc_vector)
                    logs.append(("mean_acc", mean_acc))
                    progbar.add(len(image), values=logs)
                    if finetune_encoders:
                        with warmup_scheduler.dampening():
                            scheduler.step()
                    with warmup_scheduler_fast.dampening():
                        scheduler_fast.step()
                    with warmup_scheduler_extremefast.dampening():
                        scheduler_extremefast.step()
            print(
                "Epoch [%d/%d],  Loss: %.4f, Train_Acc: %.4f,  "
                % (
                    epoch + 1,
                    custom_num_epochs,
                    np.mean(cost_vector),
                    np.mean(acc_vector),
                )
            )
            print("end training...")

        # ====================== Validation ======================
        with torch.no_grad():
            total = len(validate_dataset)
            progbar = Progbar(total, width=10, stateful_metrics=stateful_metrics)
            model.eval()
            print("begin evaluate...")
            (
                validate_acc_list,
                validate_real_items,
                validate_fake_items,
                val_loss,
                single_items,
                unimodal_preds,
                y_GT,
            ) = evaluate(
                validate_loader, model, criterion, progbar=progbar, setting=setting
            )

            validate_acc = max(validate_acc_list)
            val_thresh = validate_acc_list.index(validate_acc)

            (
                validate_real_precision,
                validate_real_recall,
                validate_real_accuracy,
                validate_real_F1,
            ) = validate_real_items
            (
                validate_fake_precision,
                validate_fake_recall,
                validate_fake_accuracy,
                validate_fake_F1,
            ) = validate_fake_items
            img_correct, text_correct, vgg_correct, ssim_correct = single_items
            img_acc, text_acc, vgg_acc, ssim_acc = (
                img_correct[val_thresh],
                text_correct[val_thresh],
                vgg_correct[val_thresh],
                ssim_correct[val_thresh],
            )

            if validate_acc > best_acc_so_far:
                best_acc_so_far = validate_acc
                best_metrics = {
                    "real_acc": validate_real_accuracy[val_thresh],
                    "fake_acc": validate_fake_accuracy[val_thresh],
                    "real_f1": validate_real_F1[val_thresh],
                    "fake_f1": validate_fake_F1[val_thresh],
                    "real_recall": validate_real_recall[val_thresh],
                    "fake_recall": validate_fake_recall[val_thresh],
                    "real_precision": validate_real_precision[val_thresh],
                    "fake_precision": validate_fake_precision[val_thresh],
                    "thresh_idx": val_thresh,
                }
                best_epoch_record = epoch + 1

            # check best single modalities
            if img_acc > best_img_acc:
                best_img_acc = img_acc
                best_img_metrics = classification_report(
                    np.concatenate(unimodal_preds[0][val_thresh]),
                    y_GT.squeeze(),
                    digits=3,
                )
            if text_acc > best_text_acc:
                best_text_acc = img_acc
                best_text_metrics = classification_report(
                    np.concatenate(unimodal_preds[1][val_thresh]),
                    y_GT.squeeze(),
                    digits=3,
                )

            print(
                "Epoch [%d/%d],  Val_Acc: %.4f. at thresh %.4f (so far %.4f in Epoch %d) ."
                % (
                    epoch + 1,
                    custom_num_epochs,
                    validate_acc,
                    val_thresh,
                    best_acc_so_far,
                    best_epoch_record,
                )
            )
            print(
                f"Single Modalities Accuracy: Img {img_acc} Text {text_acc} VGG {vgg_acc} SSIM {ssim_acc}"
            )
            print("------Real News -----------")
            print("Precision: {}".format(validate_real_precision))
            print("Recall: {}".format(validate_real_recall))
            print("Accuracy: {}".format(validate_real_accuracy))
            print("F1: {}".format(validate_real_F1))
            print("------Fake News -----------")
            print("Precision: {}".format(validate_fake_precision))
            print("Recall: {}".format(validate_fake_recall))
            print("Accuracy: {}".format(validate_fake_accuracy))
            print("F1: {}".format(validate_fake_F1))
            print("---------------------------")
            print("end evaluate...")
            print(args)
            val_losses.append(val_loss)
            if validate_acc > best_validate_acc and setting["val"] == 0:
                best_validate_acc = validate_acc
                if not os.path.exists(args.output_file):
                    os.mkdir(args.output_file)
                best_validate_dir = "./checkpoints/{}/{}_{}{}_{}.pkl".format(
                    setting["train_dataname"],
                    str(epoch + 1),
                    str(datetime.datetime.now().month),
                    str(datetime.datetime.now().day),
                    int(best_validate_acc * 100),
                )
                torch.save(model.state_dict(), best_validate_dir)
                print("Model saved at {}".format(best_validate_dir))

    with open(f"./results-{args.train_dataset}.log", "a") as f:
        if setting["val"] == 0:
            f.write(
                f"==================== {datetime.datetime.now()} ===================\n \n"
            )
            f.write(f"val_acc: {best_acc_so_far}\n")
            f.write(f"Detailed performance: {best_metrics}\n")
            f.write(f"Model checkpoint: {best_validate_dir}\n")
            f.write(f"IMG report: {best_img_metrics}\n")
            f.write(f"TEXT report: {best_text_metrics}\n")
            f.write(f"args: {args}\n \n")



def evaluate(validate_loader, model, criterion, progbar=None, setting={}):
    model.eval()
    val_loss = 0
    threshold = setting["thresh"]  
    THRESH = [threshold]

    print(f"thresh: {THRESH}")
    realnews_TP, realnews_TN, realnews_FP, realnews_FN = (
        [0] * len(THRESH),
        [0] * len(THRESH),
        [0] * len(THRESH),
        [0] * len(THRESH),
    )
    fakenews_TP, fakenews_TN, fakenews_FP, fakenews_FN = (
        [0] * len(THRESH),
        [0] * len(THRESH),
        [0] * len(THRESH),
        [0] * len(THRESH),
    )
    realnews_sum, fakenews_sum = [0] * len(THRESH), [0] * len(THRESH)
    img_correct, ssim_correct, text_correct, vgg_correct = (
        [0] * len(THRESH),
        [0] * len(THRESH),
        [0] * len(THRESH),
        [0] * len(THRESH),
    )
    y_pred_full, y_GT_full = None, []
    image_no, results = 0, []

    tsnef = torch.zeros(1, 64).cuda()
    tsnei = torch.zeros(1, 64).cuda()
    tsnet = torch.zeros(1, 64).cuda()
    all_labels = torch.zeros(1, 1).cuda()
    img_preds = [[] for i in range(len(THRESH))]
    text_preds = [[] for i in range(len(THRESH))]
    for i, items in enumerate(validate_loader):
        texts, others, clip_inputs, GT_path = items
        input_ids, attention_mask, token_type_ids = texts
        image, image_aug, labels, category, sents = others
        (
            input_ids,
            attention_mask,
            token_type_ids,
            image,
            image_aug,
            labels,
            category,
            clip_inputs,
        ) = (
            to_var(input_ids),
            to_var(attention_mask),
            to_var(token_type_ids),
            to_var(image),
            to_var(image_aug),
            to_var(labels),
            to_var(category),
            clip_inputs.to(args.device),
        )

        (
            mix_output,
            image_only_output,
            text_only_output,
            features,
            dispatch_vec
        ) = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            image=image,
            clip_inputs=clip_inputs,
            return_features=True,
        )
        (
            final_feature_main_task,
            shared_image_feature,
            shared_text_feature,
        ) = features
        tsnef = torch.cat([tsnef, final_feature_main_task], 0)
        tsnei = torch.cat([tsnei, shared_image_feature], 0)
        tsnet = torch.cat([tsnet, shared_text_feature], 0)

        if setting["is_use_bce"]:
            labels = labels.float().unsqueeze(1)
        all_labels = torch.cat([all_labels, labels], 0)
        val_loss = criterion(mix_output, labels)
        val_img_loss = criterion(image_only_output, labels)
        val_text_loss = criterion(text_only_output, labels)
        if progbar is not None:
            logs = []
            logs.append(("mix_loss", val_loss.item()))
            logs.append(("image_loss", val_img_loss.item()))
            logs.append(("text_loss", val_text_loss.item()))
            progbar.add(len(image), values=logs)

        mix_output, image_only_output, text_only_output = (
            torch.sigmoid(mix_output),
            torch.sigmoid(image_only_output),
            torch.sigmoid(text_only_output),
        )

        y_GT_full.append(labels.int().cpu().numpy())
        for thresh_idx, thresh in enumerate(THRESH):
            # _, validate_argmax = torch.max(validate_outputs, 1)
            validate_argmax = torch.where(mix_output < thresh, 0, 1)
            # validate_ssim_argmax = torch.where(aux_output < 0.5, 0, 1)
            validate_img_argmax = torch.where(image_only_output < thresh, 0, 1)
            validate_text_argmax = torch.where(text_only_output < thresh, 0, 1)
            # validate_vgg_argmax = torch.where(vgg_only_output < thresh, 0, 1)
            y_pred = (
                validate_argmax.squeeze().cpu().numpy()
            )  # y_pred = torch.tensor([0, 1, 0, 0])
            y_pred_img = validate_img_argmax.squeeze().cpu().numpy()
            img_preds[thresh_idx].append(y_pred_img)

            y_pred_text = validate_text_argmax.squeeze().cpu().numpy()
            text_preds[thresh_idx].append(y_pred_text)
            y_GT = labels.int().cpu().numpy()  # y_true=torch.tensor([0, 1, 0, 1])

            for idx, _ in enumerate(y_pred):
                if thresh_idx == 0:
                    record = {}
                    # record["final_feature"] = (
                    #     final_feature_main_task[idx].cpu().numpy().tolist()
                    # )
                    # record["image_feature"] = (
                    #     shared_image_feature[idx].cpu().numpy().tolist()
                    # )
                    # record["text_feature"] = (
                    #     shared_text_feature[idx].cpu().numpy().tolist()
                    # )
                    # record["text_feature"] = (
                    #     shared_text_feature[idx].cpu().numpy().tolist()
                    # )
                    record["dispatch_vec"] = dispatch_vec[idx].cpu().numpy().tolist()
                    record["image_no"], record["text"] = image_no, sents[idx]
                    record["y_GT"], record["y_pred"] = y_GT[idx], mix_output[idx].item()
                    (
                        record["y_pred_img"],
                        record["y_pred_text"],
                    ) = (
                        image_only_output[idx].item(),
                        text_only_output[idx].item(),
                    )
                    results.append(record)


                if y_pred_img[idx] == y_GT[idx]:
                    img_correct[thresh_idx] += 1
                if y_pred_text[idx] == y_GT[idx]:
                    text_correct[thresh_idx] += 1

                if y_GT[idx] == 1:
                    #  FAKE NEWS RESULT
                    fakenews_sum[thresh_idx] += 1
                    if y_pred[idx] == 0:
                        fakenews_FN[thresh_idx] += 1
                        realnews_FP[thresh_idx] += 1
                    else:
                        fakenews_TP[thresh_idx] += 1
                        realnews_TN[thresh_idx] += 1
                else:
                    # REAL NEWS RESULT
                    realnews_sum[thresh_idx] += 1
                    if y_pred[idx] == 1:
                        realnews_FN[thresh_idx] += 1
                        fakenews_FP[thresh_idx] += 1
                    else:
                        realnews_TP[thresh_idx] += 1
                        fakenews_TN[thresh_idx] += 1
    tsnef = tsnef[1:, :]
    tsnei = tsnei[1:, :]
    tsnet = tsnet[1:, :]
    all_labels = all_labels[1:, :]

    tsnef = torch.cat([all_labels, tsnef], 1)
    tsnei = torch.cat([all_labels, tsnei], 1)
    tsnet = torch.cat([all_labels, tsnet], 1)

    tsnef = tsnef.cpu()
    tsnei = tsnei.cpu()
    tsnet = tsnet.cpu()
    resultf = np.array(tsnef)
    resulti = np.array(tsnei)
    resultt = np.array(tsnet)

    if setting['val'] == 1:
        # np.savetxt(f"npresultf_{args.train_dataset}.txt", resultf)
        # np.savetxt(f"npresulti_{args.train_dataset}.txt", resulti)
        # np.savetxt(f"npresultt_{args.train_dataset}.txt", resultt)

        import pandas as pd
        df = pd.DataFrame(results)
        csv_file = f"./{setting['val_dataname']}_experiment_all.csv"
        df.to_csv(csv_file)
        print(f"Csv Saved at {csv_file}")

    val_accuracy, real_accuracy, fake_accuracy, real_precision, fake_precision = (
        [0] * len(THRESH),
        [0] * len(THRESH),
        [0] * len(THRESH),
        [0] * len(THRESH),
        [0] * len(THRESH),
    )
    real_recall, fake_recall, real_F1, fake_F1 = (
        [0] * len(THRESH),
        [0] * len(THRESH),
        [0] * len(THRESH),
        [0] * len(THRESH),
    )
    for thresh_idx, _ in enumerate(THRESH):
        ssim_correct[thresh_idx] = ssim_correct[thresh_idx] / (
            realnews_sum[thresh_idx] + fakenews_sum[thresh_idx]
        )
        img_correct[thresh_idx] = img_correct[thresh_idx] / (
            realnews_sum[thresh_idx] + fakenews_sum[thresh_idx]
        )
        text_correct[thresh_idx] = text_correct[thresh_idx] / (
            realnews_sum[thresh_idx] + fakenews_sum[thresh_idx]
        )
        vgg_correct[thresh_idx] = vgg_correct[thresh_idx] / (
            realnews_sum[thresh_idx] + fakenews_sum[thresh_idx]
        )

        val_accuracy[thresh_idx] = (
            realnews_TP[thresh_idx] + realnews_TN[thresh_idx]
        ) / (
            realnews_TP[thresh_idx]
            + realnews_TN[thresh_idx]
            + realnews_FP[thresh_idx]
            + realnews_FN[thresh_idx]
        )
        real_accuracy[thresh_idx] = (realnews_TP[thresh_idx]) / realnews_sum[thresh_idx]
        fake_accuracy[thresh_idx] = (fakenews_TP[thresh_idx]) / fakenews_sum[thresh_idx]
        real_precision[thresh_idx] = realnews_TP[thresh_idx] / max(
            1, (realnews_TP[thresh_idx] + realnews_FP[thresh_idx])
        )
        fake_precision[thresh_idx] = fakenews_TP[thresh_idx] / max(
            1, (fakenews_TP[thresh_idx] + fakenews_FP[thresh_idx])
        )
        real_recall[thresh_idx] = realnews_TP[thresh_idx] / max(
            1, (realnews_TP[thresh_idx] + realnews_FN[thresh_idx])
        )
        fake_recall[thresh_idx] = fakenews_TP[thresh_idx] / max(
            1, (fakenews_TP[thresh_idx] + fakenews_FN[thresh_idx])
        )
        real_F1[thresh_idx] = (
            2
            * (real_recall[thresh_idx] * real_precision[thresh_idx])
            / max(1, (real_recall[thresh_idx] + real_precision[thresh_idx]))
        )
        fake_F1[thresh_idx] = (
            2
            * (fake_recall[thresh_idx] * fake_precision[thresh_idx])
            / max(1, (fake_recall[thresh_idx] + fake_precision[thresh_idx]))
        )

    return (
        val_accuracy,
        (real_precision, real_recall, real_accuracy, real_F1),
        (fake_precision, fake_recall, fake_accuracy, fake_F1),
        val_loss,
        (img_correct, text_correct, vgg_correct, ssim_correct),
        (img_preds, text_preds),
        np.concatenate(y_GT_full),
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-output_file", type=str, default="./output", help=""
    )  # parser.add_argument('-dataset', type=str, default='weibo', help='')
    parser.add_argument("-train_dataset", type=str, default="Twitter", help="")
    parser.add_argument("-test_dataset", type=str, default="Twitter", help="")
    parser.add_argument("-checkpoint", type=str, default="", help="")
    parser.add_argument("-device", type=str, default="cuda:1", help="cuda device")
    parser.add_argument("-finetune", type=int, default=0, help="")
    parser.add_argument("-val", type=int, default=0, help="")
    parser.add_argument("-duplicate_fake_times", type=int, default=0, help="")
    parser.add_argument("-batch_size", type=int, default=16, help="")
    parser.add_argument("-epochs", type=int, default=100, help="")
    parser.add_argument("-hidden_dim", type=int, default=512, help="")
    parser.add_argument("-embed_dim", type=int, default=32, help="")
    parser.add_argument("-vocab_size", type=int, default=25, help="")
    parser.add_argument("-text_only", type=bool, default=False, help="")
    parser.add_argument("-get_MLP_score", type=int, default=0, help="")
    parser.add_argument("-int_lr", type=float, default=1e-4, help="")
    parser.add_argument("-int_beta", type=float, default=0.7, help="")
    parser.add_argument("-agr_threshold", type=float, default=0.3, help="")
    parser.add_argument("-sem_threshold", type=float, default=0.3, help="")
    parser.add_argument("-note", type=str, default="NO NOTES", help="")
    args = parser.parse_args()
    if args.get_MLP_score > 0 and args.val == 0:
        args.val = 1

    main(args)
