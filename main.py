# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-15 14:11:08
# @Last Modified by:   Jie     @Contact: jieynlp@gmail.com
# @Last Modified time: 2017-06-26 17:21:24

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
from model import LSTMTagger
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
    word_emb_norm = False
    data.build_word_pretrain_emb(emb_file, word_emb_norm)
    data.show_data_summary()
    sys.stdout.flush()
    return data

def predict_check(pred_variable, gold_variable):
    pred = pred_variable.data.numpy().argmax(axis=1)
    gold = gold_variable.data.numpy()
    right_token = np.sum(pred == gold)
    total_token = gold.size
    return right_token, total_token

def recover_label(pred_variable, gold_variable, label_alphabet):
    pred = pred_variable.data.numpy().argmax(axis=1).tolist()
    gold = gold_variable.data.numpy().tolist()
    pred_label = [label_alphabet.get_instance(idx) for idx in pred]
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
    model.eval()
    for words, chars, label in instances:
        label = autograd.Variable(torch.LongTensor(label))
        if gpu:
            label = label.cuda()
        model.zero_grad()
        # model.hidden = model.init_hidden(gpu)
        pred_score = model([words,chars], gpu)
        if gpu:
            pred_score = pred_score.cpu()
            label = label.cpu()
        pred_label, gold_label = recover_label(pred_score, label, data.label_alphabet)
        pred_results.append(pred_label)
        gold_results.append(gold_label)
    acc, p, r, f = get_ner_fmeasure(pred_results, gold_results, "BIO")
    return acc, p, r, f
    

def train(data, gpu):
    vocabulary_size = data.word_alphabet.size()
    label_size = data.label_alphabet.size()
    EMBEDDING_DIM = data.word_emb_dim
    HIDDEN_DIM = 100
    dropout = 0.2
    lstm_layer = 1
    bilstm = True
    use_char = True
    model = LSTMTagger(data, HIDDEN_DIM, dropout, lstm_layer, bilstm, use_char, gpu)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
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
        for words, chars, label in data.train_Ids:
            instance_count += 1
            # if instance_count > 1000:
            #     continue
            label = autograd.Variable(torch.LongTensor(label))
            if gpu:
                label = label.cuda()
            model.zero_grad()
            # model.hidden = model.init_hidden(gpu)
            loss, pred_score, tag_seq = model.neg_log_likelihood([words,chars],label, gpu)
            print pred_score
            print "tagseq:", tag_seq
            print "label:", label
            # loss = loss_function(pred_score, label)
            if gpu:
                pred_score = pred_score.cpu()
                label = label.cpu()
                loss =loss.cpu()
            right, whole = predict_check(pred_score, label)
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
            optimizer.step()
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
    train_file = "data/eng.train"
    dev_file = "data/eng.dev"
    test_file = "data/eng.test"
    emb_file = "data/SENNA.emb"
    gpu = torch.cuda.is_available()
    print "GPU available:", gpu
    data = data_initialization(train_file, dev_file, test_file, emb_file)
    train(data, gpu)
