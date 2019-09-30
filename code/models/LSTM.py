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

class LSTM(nn.Module):
	def __init__(self, config):
		super(LSTM, self).__init__()
		self.config = config

		word_vec_size = config.data_word_vec.shape[0]
		self.word_emb = nn.Embedding(word_vec_size, config.data_word_vec.shape[1])
		self.word_emb.weight.data.copy_(torch.from_numpy(config.data_word_vec))
		self.word_emb.weight.requires_grad = False

		# self.char_emb = nn.Embedding(config.data_char_vec.shape[0], config.data_char_vec.shape[1])
		# self.char_emb.weight.data.copy_(torch.from_numpy(config.data_char_vec))
		# char_dim = config.data_char_vec.shape[1]
		# char_hidden = 100
		# self.char_cnn = nn.Conv1d(char_dim,  char_hidden, 5)
		self.coref_embed = nn.Embedding(config.max_length, config.coref_size, padding_idx=0)
		self.ner_emb = nn.Embedding(7, config.entity_type_size, padding_idx=0)

		input_size = config.data_word_vec.shape[1] + config.coref_size + config.entity_type_size #+ char_hidden
		hidden_size = 128


		self.rnn = EncoderLSTM(input_size, hidden_size, 1, True, False, 1 - config.keep_prob, False)
		self.linear_re = nn.Linear(hidden_size, hidden_size)  # *4 for 2layer

		self.bili = torch.nn.Bilinear(hidden_size+config.dis_size, hidden_size+config.dis_size, config.relation_num)



		self.dis_embed = nn.Embedding(20, config.dis_size, padding_idx=10)



	def forward(self, context_idxs, pos, context_ner, context_char_idxs, context_lens, h_mapping, t_mapping,
				relation_mask, dis_h_2_t, dis_t_2_h):
		# para_size, char_size, bsz = context_idxs.size(1), context_char_idxs.size(2), context_idxs.size(0)
		# context_ch = self.char_emb(context_char_idxs.contiguous().view(-1, char_size)).view(bsz * para_size, char_size, -1)
		# context_ch = self.char_cnn(context_ch.permute(0, 2, 1).contiguous()).max(dim=-1)[0].view(bsz, para_size, -1)

		sent = torch.cat([self.word_emb(context_idxs) , self.coref_embed(pos), self.ner_emb(context_ner)], dim=-1)
		# sent = torch.cat([self.word_emb(context_idxs), context_ch], dim=-1)

		# context_mask = (context_idxs > 0).float()
		context_output = self.rnn(sent, context_lens)

		context_output = torch.relu(self.linear_re(context_output))


		start_re_output = torch.matmul(h_mapping, context_output)
		end_re_output = torch.matmul(t_mapping, context_output)
		# predict_re = self.bili(start_re_output, end_re_output)

		s_rep = torch.cat([start_re_output, self.dis_embed(dis_h_2_t)], dim=-1)
		t_rep = torch.cat([end_re_output, self.dis_embed(dis_t_2_h)], dim=-1)
		predict_re = self.bili(s_rep, t_rep)



		return predict_re

