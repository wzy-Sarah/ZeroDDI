from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, SubsetRandomSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import os
import os.path as osp
import copy
from torch.optim import Adam, AdamW
import torch
import time
from .utils import softmax
from tools.logging_ import get_root_logger
import random
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import copy
import sklearn
import pickle
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D


global history
history = defaultdict(list)


def train_model(model, datasets, cfg):
    logger = get_root_logger(log_level=cfg.log_level)
    train_sampler = RandomSampler(datasets[0]) #datasets[0] is train dataset
    logger.info("The number of train instances: {}".format(len(train_sampler)))


    use_collate_fn = False
    if use_collate_fn:
        train_dataloader = DataLoader(datasets[0], sampler=train_sampler,
                                      batch_size=cfg.train_batch_size,
                                      collate_fn=datasets[0].collate_fn)
    else:
        train_dataloader = DataLoader(datasets[0], sampler=train_sampler,
                                      batch_size=cfg.train_batch_size)


    no_decay = []
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': cfg.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.001}
    ]
    optimizer = Adam(optimizer_grouped_parameters, lr=cfg.learning_rate)
    torch.autograd.set_detect_anomaly(True)

    zsl_best_model = 0
    gzsl_best_model = 0
    zsl_best_epoch = 0
    gzsl_best_epoch = 0
    best_H = 0
    best_per_acc = 0

    for epoch in range(cfg.num_epochs):
        batch_step = 0
        batch_loss = 0
        #train_dataloader.sampler.set_epoch(epoch)
        for step, batch in enumerate(train_dataloader):
            t1 = time.time()
            model.train()
            optimizer.zero_grad()
            outputs = model(batch)
            loss = outputs[0]

            loss.backward()
            optimizer.step()

            batch_loss += loss.item()
            batch_step += 1
            
        if (epoch+1)%1==0:

            logger.info(f"epoch is {epoch} || Train batch_loss is {batch_loss / batch_step} \n")
            history['train_loss'].append(batch_loss / batch_step)
            per_acc = evaluate(model, datasets[1], logger, cfg, "zsl", "test")  # zsl_val_dataset
            H = evaluate(model, datasets[2], logger, cfg, "gzsl", "test")  # gzsl_val_dataset
            if H > best_H:
                best_H = H
                gzsl_best_epoch = epoch
                gzsl_best_model = copy.deepcopy(model.state_dict())

            if per_acc > best_per_acc:
                best_per_acc = per_acc
                zsl_best_epoch = epoch
                zsl_best_model = copy.deepcopy(model.state_dict())
            # print("time",time.time()-t1)
        if (epoch + 1) % 20 == 0:
            #if torch.distributed.get_rank() == 0:
            torch.save(model.state_dict(), osp.join(osp.join(cfg.work_dir, 'model_parameter'),
                                                    f'model_epoch{epoch + 1}_seed{cfg.seednumber}.pkl'))
            torch.save(zsl_best_model, osp.join(osp.join(cfg.work_dir, 'model_parameter'),
                                                f'zsl_model_best_epoch{epoch + 1}_seed{cfg.seednumber}.pkl'))
            torch.save(gzsl_best_model, osp.join(osp.join(cfg.work_dir, 'model_parameter'),
                                                 f'gzsl_model_best_epoch{epoch + 1}_seed{cfg.seednumber}.pkl'))

    logger.info(f"The gzsl best epoch is {gzsl_best_epoch + 1}")
    logger.info(f"The zsl best epoch is {zsl_best_epoch + 1}")
    # save history



def evaluate(model, dataset, logger, cfg, zsl, aaa="test", visualize_acc=False):

    eval_sampler = RandomSampler(dataset)
    use_collate_fn = False
    evel_data_len = len(dataset)
    if use_collate_fn:
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler,
                                     batch_size=64, collate_fn=dataset.collate_fn,drop_last=True)
    else:
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler,
                                     batch_size=64,drop_last=True)
      
    mode = dataset.mode
    logger.info(
        f"The number of {mode} instances: {len(eval_sampler)}, the class has: {len(dataset.current_dataset_eventid_uni)}")
    embid2eventid = dataset.embid2eventid

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    gt_emb_ids = None
    instances = None
    prototypes = None
    model.eval()

    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(batch)
    
        loss, Logits_all, gt_id, instance, prototype = outputs[:5]  # (cost, logits, emb_ids, Matmul_gnn_W, right_output_all)
        eval_loss += loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = Logits_all.detach().cpu().numpy()
            gt_emb_ids = gt_id.detach().cpu().numpy()
            instances = instance.detach().cpu().numpy()
            prototypes = prototype.detach().cpu().numpy()
        else:
            preds = np.append(preds, Logits_all.detach().cpu().numpy(), axis=0)
            gt_emb_ids = np.append(gt_emb_ids, gt_id.detach().cpu().numpy(), axis=0)
            instances = np.append(instances, instance.detach().cpu().numpy(), axis=0)
            prototypes = np.append(prototypes, prototype.detach().cpu().numpy(), axis=0)
    eval_loss = eval_loss / nb_eval_steps
    if zsl == "zsl":
        Val_Evaluation, per_class_top_1_acc, acc_per_class_list, true_label_count = zsl_accuracy(preds, gt_emb_ids)


        logger.info(f"*****Begin to {mode}, eval loss: {eval_loss}****** ")
        logger.info(
            "per_class_top@1_acc:%f , top@1 acc:%f, top@2 acc:%f, top@3 acc:%f, top@5 acc:%f" % (
                per_class_top_1_acc,
                Val_Evaluation["Accuracy"],
                Val_Evaluation["Top2Acc"],
                Val_Evaluation["Top3Acc"],
                Val_Evaluation["Top5Acc"]))
        logger.info("************************\n")

    
        return per_class_top_1_acc

    elif zsl == "gzsl":
        # seen_labels, embedid2eventid, H = model.evaluate(mode, preds, gt_ids, instances, prototypes,logger,zsl)
        Val_Evaluation, top1_acc_seen, top1_acc_unseen,top1H, per_class_top_1_acc, acc_seen, \
        acc_unseen, H, two_classify_seenacc, \
        two_classify_unseenacc, bi_acc, acc_per_class_list_seen, true_label_count_seen, \
        acc_per_class_list_unseen, true_label_count_unseen = gzsl_accuracy(preds, gt_emb_ids, embid2eventid,
                                                                           cfg.model.seen_labels)



        logger.info(f"*****Begin to {mode}, eval loss: {eval_loss}****** ")
        logger.info(
            "per_class_top@1_acc:%f , top@1 acc:%f, top@2 acc:%f, top@3 acc:%f, "
            "top@5 acc:%f, top1_acc_seen:%f, top1_acc_unseen: %f, top1_H: %f, "
            "per_seenacc:%f, per_unseenacc:%f, H:%f, two_classify_seenacc:%f, "
            "two_classify_unseenacc:%f, bi_acc:%f" % (
                per_class_top_1_acc,
                Val_Evaluation["Accuracy"],
                Val_Evaluation["Top2Acc"],
                Val_Evaluation["Top3Acc"],
                Val_Evaluation["Top5Acc"],
                top1_acc_seen,
                top1_acc_unseen,
                top1H,
                acc_seen,
                acc_unseen,
                H,
                two_classify_seenacc,
                two_classify_unseenacc,
                bi_acc))
        logger.info("************************\n")

        return H




def zsl_accuracy(logits, ids):
    probabilities = softmax(logits, axis=1)

    get_top_n = (-probabilities).argsort()[:, 0:10]
    preds = np.argmax(logits, axis=1)

    Val_Evaluation = GetAccuracy(preds, get_top_n, ids)
    per_class_top_1_acc, acc_per_class_list, true_label_count = average_per_class_top_1_acc(preds, ids)
    return Val_Evaluation, per_class_top_1_acc, acc_per_class_list, true_label_count


def gzsl_accuracy(logits, ids, embedid2eventid, seen_labels):
    probabilities = softmax(logits, axis=1)

    get_top_n = (-probabilities).argsort()[:, 0:10]
    preds = np.argmax(logits, axis=1)

    Val_Evaluation = GetAccuracy(preds, get_top_n, ids)
    per_class_top_1_acc, _, _ = average_per_class_top_1_acc(preds, ids)

    # change
    true_eventid_list = []
    for i in range(ids.shape[0]):
        true_eventid_list.append(embedid2eventid[ids[i]])

    pred_top1_id = list(np.squeeze((-probabilities).argsort()[:, 0]))
    pred_eventid_list = []
    for i in pred_top1_id:
        pred_eventid_list.append(embedid2eventid[i])
    true_eventid = np.array(true_eventid_list)
    pred_eventid = np.array(pred_eventid_list)

    # mask the train triplet
    mask_seen = [index for index, tri in enumerate(true_eventid) if tri in seen_labels]
    mask_unseen = [index for index, tri in enumerate(true_eventid) if tri not in seen_labels]

    pred_seen = pred_eventid[mask_seen]
    true_seen = true_eventid[mask_seen]
    pred_unseen = pred_eventid[mask_unseen]
    true_unseen = true_eventid[mask_unseen]

    per_acc_seen, acc_per_class_list_seen, true_label_count_seen = average_per_class_top_1_acc(pred_seen, true_seen)
    per_acc_unseen, acc_per_class_list_unseen, true_label_count_unseen = average_per_class_top_1_acc(pred_unseen,
                                                                                                     true_unseen)
    match_seen = np.equal(pred_seen, true_seen)
    match_unseen = np.equal(pred_unseen, true_unseen)
    top1_acc_seen = np.sum(match_seen) / len(mask_seen)
    top1_acc_unseen = np.sum(match_unseen) / len(mask_unseen)
    top1H = 2 * (top1_acc_seen * top1_acc_unseen) / (top1_acc_seen + top1_acc_unseen)

    H = 2 * (per_acc_seen * per_acc_unseen) / (per_acc_seen + per_acc_unseen)

    # Binary classification
    true_seen_eventid = set(list(true_seen))
    true_unseen_eventid = set(list(true_unseen))
    gt_twoclassify_label = []
    pred_twoclassify_label = []
    for i in true_eventid_list:
        if i in true_seen_eventid:
            gt_twoclassify_label.append(
                1)  # if true eventid belong to seen class, 1; elif true eventid belong to unseen class, 0
        elif i in true_unseen_eventid:
            gt_twoclassify_label.append(0)
    assert len(gt_twoclassify_label) == len(true_eventid_list)

    for i in pred_eventid_list:
        if i in true_seen_eventid:
            pred_twoclassify_label.append(
                1)  # if true eventid belong to seen class, 1; elif true eventid belong to unseen class, 0
        elif i in true_unseen_eventid:
            pred_twoclassify_label.append(0)
        else:
            pred_twoclassify_label.append(9)
    assert len(pred_twoclassify_label) == len(pred_eventid_list)

    two_classify_matchseen = np.equal(np.array(gt_twoclassify_label)[mask_seen],
                                      np.array(pred_twoclassify_label)[mask_seen])
    two_classify_seenacc = np.sum(two_classify_matchseen != 0) / len(mask_seen)
    two_classify_matchunseen = np.equal(np.array(gt_twoclassify_label)[mask_unseen],
                                        np.array(pred_twoclassify_label)[mask_unseen])

    two_classify_unseenacc = np.sum(two_classify_matchunseen != 0) / len(mask_unseen)
    bi_acc = np.sum(np.equal(np.array(gt_twoclassify_label),
                             np.array(pred_twoclassify_label)) != 0) / len(gt_twoclassify_label)

    return Val_Evaluation, top1_acc_seen, top1_acc_unseen, top1H, per_class_top_1_acc, per_acc_seen, per_acc_unseen, H, \
           two_classify_seenacc, two_classify_unseenacc, bi_acc, acc_per_class_list_seen, true_label_count_seen, \
           acc_per_class_list_unseen, true_label_count_unseen


def GetAccuracy(Y_Pred, top_10_classes, Y_True):
    """ Returns Accuracy when multi-label are provided for each instance. It will be counted true if predicted y is among the true labels
    Args:
        Y_Pred (int array): the predicted labels
        Probabilities (float [][] array): the probabilities predicted for each class for each instance
        Y_True (int[] array): the true labels, for each instance it should be a list
    """

    def intersection(lst1, lst2):
        return list(set(lst1) & set(lst2))

    count_true = 0
    count_true_2 = 0
    count_true_3 = 0
    count_true_5 = 0
    count_true_10 = 0
    for i in range(len(Y_Pred)):
        if Y_Pred[i] == Y_True[i]:
            count_true += 1
        if len(intersection(top_10_classes[i], [Y_True[i]])) > 0:
            count_true_10 += 1
        if len(intersection(top_10_classes[i][:5], [Y_True[i]])) > 0:
            count_true_5 += 1
        if len(intersection(top_10_classes[i][:3], [Y_True[i]])) > 0:
            count_true_3 += 1
        if len(intersection(top_10_classes[i][:2], [Y_True[i]])) > 0:
            count_true_2 += 1

    Evaluations = {"Accuracy": (float(count_true) / len(Y_Pred)),
                   "Top2Acc": (float(count_true_2) / len(Y_Pred)),
                   "Top3Acc": (float(count_true_3) / len(Y_Pred)),
                   "Top5Acc": (float(count_true_5) / len(Y_Pred))}
    return Evaluations


def average_per_class_top_1_acc(preds, batch_true_label_):
    """
    Zero-Shot Learning - The Good, the Bad and the Ugly
    """
    split_class = defaultdict(list)
    true_label_count = defaultdict(list)

    for i, label in enumerate(batch_true_label_):
        split_class[label].append(preds[i])
        true_label_count[label].append(1)

    acc_per_class_list = []
    ids = list(split_class.keys())

    for key, values in split_class.items():
        length = len(values)
        count = 0
        for v in values:
            if v == key:
                count += 1
        acc_per_class = count / length
        acc_per_class_list.append(acc_per_class)
    per_class_top_1_acc = np.mean(np.array(acc_per_class_list))
    return per_class_top_1_acc, acc_per_class_list, true_label_count



