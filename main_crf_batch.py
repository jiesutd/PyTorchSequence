# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-15 14:11:08
# @Last Modified by:   Jie     @Contact: jieynlp@gmail.com
# @Last Modified time: 2017-07-02 22:47:51

import time
import sys
import random
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils.metric import get_ner_fmeasure
from model_crf_batch import LSTMTagger
from utils.data import Data

random.seed(100)
torch.manual_seed(100)
np.random.seed(100)

def data_initialization(train_file, dev_file, test_file, emb_file):
    data = Data()
    data.number_normalized = True
    data.build_alphabet(train_file)
    data.build_alphabet(dev_file)
    data.build_alphabet(test_file)
    data.fix_alphabet()
    data.generate_instance(train_file,'train')
    data.generate_instance(dev_file,'dev')
    data.generate_instance(test_file,'test')
    data.build_word_pretrain_emb(emb_file)
    return data

def predict_check(pred_tag, gold_variable):
    gold = gold_variable.data.numpy()
    right_token = np.sum(pred_tag == gold)
    total_token = gold.size
    return right_token, total_token

def recover_label(pred_tag, gold_variable, label_alphabet):
    # pred = pred_variable.data.numpy().argmax(axis=1).tolist()
    gold = gold_variable.data.numpy().tolist()
    pred_label = [label_alphabet.get_instance(idx) for idx in pred_tag]
    gold_label = [label_alphabet.get_instance(idx) for idx in gold]
    return pred_label, gold_label


def evaluate(data, model, name, gpu):
    if name == "dev":
        instances = data.dev_Ids
    elif name == "test":
        instances = data.test_Ids
    else:
        print "Error: wrong evaluate name,", name
    right_token = 0
    whole_token = 0
    pred_results = []
    gold_results = []
    ## set model in eval model
    model.eval()
    for words, chars, label in instances:
        label = autograd.Variable(torch.LongTensor(label))
        pred_score, tag_seq = model([words,chars], gpu)
        pred_label, gold_label = recover_label(tag_seq, label, data.label_alphabet)
        pred_results.append(pred_label)
        gold_results.append(gold_label)
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)
    return acc, p, r, f
    

def train(data, gpu):
    data.HP_gpu = gpu
    data.show_data_summary()
    model = LSTMTagger(data)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=data.HP_lr, momentum=data.HP_momentum)
    best_dev = -1
    ## start training
    for idx in range(100):
        epoch_start = time.time()
        temp_start = epoch_start
        print "Epoch:", idx
        instance_count = 0
        sample_id = 0
        sample_loss = 0
        right_token = 0
        whole_token = 0
        random.shuffle(data.train_Ids)
        ## set model in train model
        model.train()
        model.zero_grad()
        for words, chars, label in data.train_Ids:
            instance_count += 1
            # if instance_count > 500:
            #     continue
            label = autograd.Variable(torch.LongTensor(label))
            loss,_, tag_seq = model._get_neg_log_likilihood_loss([words,chars], label, gpu)
            right, whole = predict_check(tag_seq, label)
            right_token += right
            whole_token += whole
            sample_loss += loss.data.numpy()[0]
            if instance_count%500 == 0:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f"%(instance_count, temp_cost, sample_loss, right_token, whole_token,(right_token+0.)/whole_token))
                sys.stdout.flush()
                sample_loss = 0
            loss.backward()
            ## manual set batch size
            if instance_count%data.HP_batch_size == 0:
                ## clip gradient
                # torch.nn.utils.clip_grad_norm(model.parameters(), data.HP_clip)
                optimizer.step()
                model.zero_grad()
            
            
                
        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print("Epoch: %s training finished. Time: %.2fs"%(idx, epoch_cost))
        acc, p, r, f = evaluate(data, model, "dev", gpu)
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish
        print("Dev: time:%.2fs; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(dev_cost, acc, p, r, f))
        if f > best_dev:
            print "Exceed best f, previous best f:", best_dev
            best_dev = f 
        # ## decode test
        acc, p, r, f = evaluate(data, model, "test", gpu)
        test_finish = time.time()
        test_cost = test_finish - dev_finish
        print("Test: time: %.2fs; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(test_cost, acc, p, r, f))


if __name__ == '__main__':
    BIO = False
    emb = "GLOVE"
    if BIO:    
        train_file = "data/train.bio"
        dev_file = "data/dev.bio"
        test_file = "data/test.bio"
    else:
        train_file = "data/train.bmes"
        dev_file = "data/dev.bmes"
        test_file = "data/test.bmes"
    if emb == "SENNA":
        emb_file = "data/SENNA.emb"
    elif emb == "GLOVE":
        emb_file = "data/glove.6B.100d.txt"
    gpu = torch.cuda.is_available()
    print "GPU available:", gpu
    print "Word Embedding:", emb
    data = data_initialization(train_file, dev_file, test_file, emb_file)
    train(data, gpu)
