import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from CRF import CRF
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

class LSTMTagger(nn.Module):
    def __init__(self, data,  gpu = False):
        super(LSTMTagger, self).__init__()
        self.use_char = data.HP_use_char
        self.batch_size = data.HP_batch_size
        if self.use_char:
            self.char_embedding_dim = 30
            self.char_hidden_dim = 25
            self.char_drop = nn.Dropout(data.HP_dropout)
            self.char_embeddings = nn.Embedding(data.char_alphabet.size(), self.char_embedding_dim)
            self.char_bilstm_flag = True
            self.char_lstm_layer = 1
            self.char_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(data.char_alphabet.size(), self.char_embedding_dim)))
            self.char_lstm = nn.LSTM(self.char_embedding_dim, self.char_hidden_dim, num_layers=self.char_lstm_layer, batch_first=False, bidirectional=self.char_bilstm_flag)
            self.char_hidden = self.init_char_hidden(data.HP_gpu)
            if data.HP_gpu:
                self.char_drop = self.char_drop.cuda()
                self.char_embeddings = self.char_embeddings.cuda()
                self.char_lstm = self.char_lstm.cuda()
        self.embedding_dim = data.word_emb_dim
        self.hidden_dim = data.HP_hidden_dim
        self.drop = nn.Dropout(data.HP_dropout)
        self.word_embeddings = nn.Embedding(data.word_alphabet.size(), self.embedding_dim)
        self.bilstm_flag = data.HP_bilstm
        self.lstm_layer = data.HP_lstm_layer
        if data.pretrain_word_embedding is not None:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(data.pretrain_word_embedding))
        else:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(data.word_alphabet.size(), self.embedding_dim)))
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        if data.HP_use_char:
            self.lstm = nn.LSTM(self.embedding_dim+2*self.char_hidden_dim, data.HP_hidden_dim, num_layers=self.lstm_layer, batch_first=False, bidirectional=self.bilstm_flag)
        else:
            self.lstm = nn.LSTM(self.embedding_dim, data.HP_hidden_dim, num_layers=self.lstm_layer, batch_first=False, bidirectional=self.bilstm_flag)

        # The linear layer that maps from hidden state space to tag space
        if self.bilstm_flag:
            self.hidden2tag = nn.Linear(2*data.HP_hidden_dim, data.label_alphabet.size()+2)
        else:
            self.hidden2tag = nn.Linear(self.hidden_dim, data.label_alphabet.size()+2)
        self.hidden = self.init_hidden(data.HP_gpu)

        if data.HP_gpu:
            self.drop = self.drop.cuda()
            self.word_embeddings = self.word_embeddings.cuda()
            self.lstm = self.lstm.cuda()
            self.hidden2tag = self.hidden2tag.cuda()
        self.CRF = CRF(data.label_alphabet_size)

        


    def init_hidden(self, gpu):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        
        if self.bilstm_flag:
            h0 = autograd.Variable(torch.zeros(2*self.lstm_layer, 1, self.hidden_dim))
            c0 = autograd.Variable(torch.zeros(2*self.lstm_layer, 1, self.hidden_dim))
        else:
            h0 = autograd.Variable(torch.zeros(self.lstm_layer, 1, self.hidden_dim))
            c0 = autograd.Variable(torch.zeros(self.lstm_layer, 1, self.hidden_dim))
        if gpu:
            h0 = h0.cuda()
            c0 = c0.cuda()
        return (h0,c0)

    def init_char_hidden(self, gpu):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        
        if self.char_bilstm_flag:
            h0 = autograd.Variable(torch.zeros(2*self.char_lstm_layer, 1, self.char_hidden_dim))
            c0 = autograd.Variable(torch.zeros(2*self.char_lstm_layer, 1, self.char_hidden_dim))
        else:
            h0 = autograd.Variable(torch.zeros(self.char_lstm_layer, 1, self.char_hidden_dim))
            c0 = autograd.Variable(torch.zeros(self.char_lstm_layer, 1, self.char_hidden_dim))
        if gpu:
            h0 = h0.cuda()
            c0 = c0.cuda()
        return (h0,c0)


    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def _get_lstm_features(self, inputs, gpu=False):
        [words, chars] = inputs
        sentence_length = len(words)
        words = autograd.Variable(torch.LongTensor(words))
        if gpu:
            words = words.cuda()
        if self.use_char:
            ## calculate char lstm
            char_outs = []
            for idx in range(sentence_length):
                word_length = len(chars[idx])         
                char = autograd.Variable(torch.LongTensor(chars[idx]))
                if gpu:
                    char = char.cuda()
                char_embeds = self.char_drop(self.char_embeddings(char))
                char_embeds = char_embeds.view(word_length, 1, -1)
                self.char_hidden = self.init_char_hidden(gpu)
                char_rnn_out, self.char_hidden = self.char_lstm(char_embeds, self.char_hidden)
                char_outs.append(self.char_hidden[0].view(1,-1))
            char_rnn = torch.cat(char_outs).view(sentence_length,1,-1)
            ## lookup word embeddings
            embeds = self.drop(self.word_embeddings(words))
            embeds = embeds.view(sentence_length, 1, -1)
            ## concat word and char together
            word_reps = torch.cat([embeds, char_rnn], 2)
        else:
            embeds = self.drop(self.word_embeddings(words))
            word_reps = embeds.view(sentence_length, 1, -1)
        self.hidden = self.init_hidden(gpu)
        lstm_out, self.hidden = self.lstm(word_reps, self.hidden)
        feats = self.hidden2tag(lstm_out.view(sentence_length, -1))
        if gpu:
            feats = feats.cpu()
        return feats
    

    def _get_neg_log_likilihood_loss(self, inputs, tags, gpu):
        feats = self._get_lstm_features(inputs, gpu)
        loss = self.CRF._get_neg_log_likilihood_loss(feats, tags)
        score, tag_seq = self.CRF._viterbi_decode(feats)
        return loss, score, tag_seq


    def forward(self, inputs, gpu=False):
        feats = self._get_lstm_features(inputs, gpu)
        # tag_scores = F.log_softmax(feats)
        score, tag_seq = self.CRF._viterbi_decode(feats)
        return score, tag_seq
        