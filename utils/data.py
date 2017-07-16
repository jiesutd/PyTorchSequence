# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-14 17:34:32
# @Last Modified by:   Jie     @Contact: jieynlp@gmail.com
# @Last Modified time: 2017-07-02 21:14:17
import sys
import numpy as np
from alphabet import Alphabet
from functions import *

START = "</s>"
UNKNOWN = "</unk>"
PADDING = "</pad>"

class Data:
    def __init__(self):
        self.MAX_SENTENCE_LENGTH = 512
        self.MAX_WORD_LENGTH = -1
        self.number_normalized = False
        self.word_alphabet = Alphabet('word')
        self.char_alphabet = Alphabet('character')
        self.word_alphabet.add(START)
        self.word_alphabet.add(UNKNOWN)
        self.char_alphabet.add(START)
        self.char_alphabet.add(UNKNOWN)
        self.char_alphabet.add(PADDING)
        self.label_alphabet = Alphabet('label')
        self.tagScheme = "NoSeg"

        self.train_texts = []
        self.dev_texts = []
        self.test_texts = []
        self.raw_texts = []

        self.train_Ids = []
        self.dev_Ids = []
        self.test_Ids = []
        self.raw_Ids = []

        self.word_emb_dim = 50
        self.pretrain_word_embedding = None
        self.label_size = 0
        self.word_alphabet_size = 0
        self.char_alphabet_size = 0
        self.label_alphabet_size = 0
        ### hyperparameters
        self.HP_batch_size = 10
        self.HP_hidden_dim = 200
        self.HP_dropout = 0.5
        self.HP_lstm_layer = 1
        self.HP_bilstm = True
        self.HP_use_char = True
        self.HP_gpu = False
        self.HP_lr = 0.015
        self.HP_lr_decay = 0
        self.HP_clip = 5.0
        self.HP_momentum = 0

        
    def show_data_summary(self):
        print("DATA SUMMARY START:")
        print("     Tag          scheme: %s"%(self.tagScheme))
        print("     MAX SENTENCE LENGTH: %s"%(self.MAX_SENTENCE_LENGTH))
        print("     MAX   WORD   LENGTH: %s"%(self.MAX_WORD_LENGTH))
        print("     Number   normalized: %s"%(self.number_normalized))
        print("     Word  alphabet size: %s"%(self.word_alphabet_size))
        print("     Char  alphabet size: %s"%(self.char_alphabet_size))
        print("     Label alphabet size: %s"%(self.label_alphabet_size))
        print("     Word embedding size: %s"%(self.word_emb_dim))
        print("     Train instance number: %s"%(len(self.train_texts)))
        print("     Dev   instance number: %s"%(len(self.dev_texts)))
        print("     Test  instance number: %s"%(len(self.test_texts)))
        print("     Raw   instance number: %s"%(len(self.raw_texts)))
        print("     Hyperpara  batch size: %s"%(self.HP_batch_size))
        print("     Hyperpara          lr: %s"%(self.HP_lr))
        print("     Hyperpara    lr_decay: %s"%(self.HP_lr_decay))
        print("     Hyperpara     HP_clip: %s"%(self.HP_clip))
        print("     Hyperpara    momentum: %s"%(self.HP_momentum))
        print("     Hyperpara  hidden_dim: %s"%(self.HP_hidden_dim))
        print("     Hyperpara     dropout: %s"%(self.HP_dropout))
        print("     Hyperpara  lstm_layer: %s"%(self.HP_lstm_layer))
        print("     Hyperpara      bilstm: %s"%(self.HP_bilstm))
        print("     Hyperpara    use_char: %s"%(self.HP_use_char))
        print("     Hyperpara         GPU: %s"%(self.HP_gpu))
        print("DATA SUMMARY END.")
        sys.stdout.flush()


    def build_alphabet(self, input_file):
        in_lines = open(input_file,'r').readlines()
        for line in in_lines:
            if len(line) > 2:
                pairs = line.strip().split()
                word = pairs[0]
                if self.number_normalized:
                    word = normalize_word(word)
                label = pairs[-1]
                self.label_alphabet.add(label)
                self.word_alphabet.add(word)
                for char in word:
                    self.char_alphabet.add(char)
        self.word_alphabet_size = self.word_alphabet.size()
        self.char_alphabet_size = self.char_alphabet.size()
        self.label_alphabet_size = self.label_alphabet.size()
        startS = False
        startB = False
        for label,_ in self.label_alphabet.iteritems():
            if "S-" in label.upper():
                startS = True
            elif "B-" in label.upper():
                startB = True
        if startB:
            if startS:
                self.tagScheme = "BMES"
            else:
                self.tagScheme = "BIO"


    def fix_alphabet(self):
        self.word_alphabet.close()
        self.char_alphabet.close()
        self.label_alphabet.close()

    def build_word_pretrain_emb(self, emb_path, norm=False):
        self.pretrain_word_embedding, self.word_emb_dim = build_pretrain_embedding(emb_path, self.word_alphabet, self.word_emb_dim, norm)


    def generate_instance(self, input_file, name):
        self.fix_alphabet()
        if name == "train":
            self.train_texts, self.train_Ids = read_instance(input_file, self.word_alphabet, self.char_alphabet, self.label_alphabet, self.number_normalized, self.MAX_WORD_LENGTH)
        elif name == "dev":
            self.dev_texts, self.dev_Ids = read_instance(input_file, self.word_alphabet, self.char_alphabet, self.label_alphabet, self.number_normalized, self.MAX_WORD_LENGTH)
        elif name == "test":
            self.test_texts, self.test_Ids = read_instance(input_file, self.word_alphabet, self.char_alphabet, self.label_alphabet, self.number_normalized, self.MAX_WORD_LENGTH)
        else:
            print("Error: you can only generate train/dev/test instance! Illegal input:%s"%(name))






