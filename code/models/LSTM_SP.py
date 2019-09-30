import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch import nn
import numpy as np
import math
from torch.nn import init
from torch.nn.utils import rnn
from layers import EncoderLSTM


class LSTM_SP(nn.Module):
	def __init__(self, config):
		super(LSTM_SP, self).__init__()
		self.config = config

		word_vec_size = config.data_word_vec.shape[0]
		self.word_emb = nn.Embedding(word_vec_size, config.data_word_vec.shape[1])
		self.word_emb.weight.data.copy_(torch.from_numpy(config.data_word_vec))
		self.word_emb.weight.requires_grad = False

		# char_size = config.data_char_vec.shape[0]
		# self.char_emb = nn.Embedding(char_size, config.data_char_vec.shape[1])
		# self.char_emb.weight.data.copy_(torch.from_numpy(config.data_char_vec))
		# char_dim = config.data_char_vec.shape[1]
		# char_hidden = 100
		# self.char_cnn = nn.Conv1d(char_dim,  char_hidden, 5)

		self.coref_embed = nn.Embedding(config.max_length, config.coref_size, padding_idx=0)
		self.ner_emb = nn.Embedding(7, config.entity_type_size, padding_idx=0)


		hidden_size = 128
		input_size = config.data_word_vec.shape[1] + config.coref_size + config.entity_type_size# + char_hidden

		self.rnn = EncoderLSTM(input_size, hidden_size, 1, True, True, 1 - config.keep_prob, False)

		self.relation_embed = nn.Embedding(config.relation_num, hidden_size, padding_idx=0)


		self.linear_t = nn.Linear(hidden_size*2, hidden_size)  # *4 for 2layer
		# self.bili = torch.nn.Bilinear(hidden_size, hidden_size, hidden_size)

		self.linear_re = nn.Linear(hidden_size*3, 1)


	def forward(self, context_idxs, pos, context_ner, context_char_idxs, context_lens, sent_h_mapping, sent_t_mapping, relation_label):
		# para_size, char_size, bsz = context_idxs.size(1), context_char_idxs.size(2), context_idxs.size(0)
		# context_ch = self.char_emb(context_char_idxs.contiguous().view(-1, char_size)).view(bsz * para_size, char_size, -1)
		# context_ch = self.char_cnn(context_ch.permute(0, 2, 1).contiguous()).max(dim=-1)[0].view(bsz, para_size, -1)

		sent = torch.cat([self.word_emb(context_idxs) , self.coref_embed(pos), self.ner_emb(context_ner)], dim=-1)

		el = sent_h_mapping.size(1)
		re_embed = (self.relation_embed(relation_label).unsqueeze(1)).expand(-1, el, -1)

		context_output = self.rnn(sent, context_lens)
		context_output = torch.relu(self.linear_t(context_output))
		start_re_output = torch.matmul(sent_h_mapping, context_output)
		end_re_output = torch.matmul(sent_t_mapping, context_output)

		sent_output = torch.cat([start_re_output, end_re_output, re_embed], dim=-1)
		predict_sent = self.linear_re(sent_output).squeeze(2)

		# predict_sent = torch.sum(self.bili(start_re_output, end_re_output)*re_embed, dim=-1)

		return predict_sent
