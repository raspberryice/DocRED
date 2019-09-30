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
from layers import EncoderLSTM


class BiLSTM(nn.Module):
	def __init__(self, config):
		super(BiLSTM, self).__init__()
		self.config = config

		# word embeddings are pretrained and freezed 
		word_vec_size = config.data_word_vec.shape[0]
		self.word_emb = nn.Embedding(word_vec_size, config.data_word_vec.shape[1])
		self.word_emb.weight.data.copy_(torch.from_numpy(config.data_word_vec))

		self.word_emb.weight.requires_grad = False

		
		# character embeddings are pretrained, this is commented out 
		# performance is similar with char_embed
		# self.char_emb = nn.Embedding(config.data_char_vec.shape[0], config.data_char_vec.shape[1])
		# self.char_emb.weight.data.copy_(torch.from_numpy(config.data_char_vec))

		# char_dim = config.data_char_vec.shape[1]
		# char_hidden = 100
		# self.char_cnn = nn.Conv1d(char_dim,  char_hidden, 5)

		hidden_size = 128
		input_size = config.data_word_vec.shape[1]

		self.use_entity_type = True
		self.use_coreference = True
		self.use_distance = True

		# entity type embeddings are learned, 7 types in total 
		if self.use_entity_type:
			input_size += config.entity_type_size
			self.ner_emb = nn.Embedding(7, config.entity_type_size, padding_idx=0)

		# entity embeddings are learned 
		if self.use_coreference:
			input_size += config.coref_size
			self.entity_embed = nn.Embedding(config.max_length, config.coref_size, padding_idx=0)


		self.rnn = EncoderLSTM(input_size, hidden_size, nlayers=1, concat=True, bidir=True, dropout=1 - config.keep_prob, return_last=False)
		self.linear_re = nn.Linear(hidden_size*2, hidden_size)

		if self.use_distance:
			self.dis_embed = nn.Embedding(20, config.dis_size, padding_idx=10)
			self.bili = torch.nn.Bilinear(hidden_size+config.dis_size, hidden_size+config.dis_size, config.relation_num)
		else:
			self.bili = torch.nn.Bilinear(hidden_size, hidden_size, config.relation_num)

	def forward(self, context_idxs, pos, context_ner, context_char_idxs, context_lens, h_mapping, t_mapping,
				relation_mask, dis_h_2_t, dis_t_2_h):
	
		sent = self.word_emb(context_idxs)
		if self.use_coreference:
			sent = torch.cat([sent, self.entity_embed(pos)], dim=-1)

		if self.use_entity_type:
			sent = torch.cat([sent, self.ner_emb(context_ner)], dim=-1)

		# the last token from biLSTM is returned 
		context_output = self.rnn(sent, context_lens)

		context_output = torch.relu(self.linear_re(context_output))


		start_re_output = torch.matmul(h_mapping, context_output)
		end_re_output = torch.matmul(t_mapping, context_output)


		if self.use_distance:
			s_rep = torch.cat([start_re_output, self.dis_embed(dis_h_2_t)], dim=-1)
			t_rep = torch.cat([end_re_output, self.dis_embed(dis_t_2_h)], dim=-1)
			predict_re = self.bili(s_rep, t_rep)
		else:
			predict_re = self.bili(start_re_output, end_re_output)

		return predict_re


