import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable 
from torch.nn.utils import rnn

class LockedDropout(nn.Module):
	def __init__(self, dropout):
		super().__init__()
		self.dropout = dropout

	def forward(self, x):
		dropout = self.dropout
		if not self.training:
			return x
		m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
		mask = Variable(m.div_(1 - dropout), requires_grad=False)
		mask = mask.expand_as(x)
		return mask * x


class EncoderLSTM(nn.Module):
	def __init__(self, input_size, num_units, nlayers, concat, bidir, dropout, return_last):
		super().__init__()
		self.rnns = []
		for i in range(nlayers):
			if i == 0:
				input_size_ = input_size
				output_size_ = num_units
			else:
				input_size_ = num_units if not bidir else num_units * 2
				output_size_ = num_units
			self.rnns.append(nn.LSTM(input_size_, output_size_, 1, bidirectional=bidir, batch_first=True))
		self.rnns = nn.ModuleList(self.rnns)

		self.init_hidden = nn.ParameterList([nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])
		self.init_c = nn.ParameterList([nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])

		self.dropout = LockedDropout(dropout)
		self.concat = concat
		self.nlayers = nlayers
		self.return_last = return_last

		# self.reset_parameters()

	def reset_parameters(self):
		for rnn in self.rnns:
			for name, p in rnn.named_parameters():
				if 'weight' in name:
					p.data.normal_(std=0.1)
				else:
					p.data.zero_()

	def get_init(self, bsz, i):
		return self.init_hidden[i].expand(-1, bsz, -1).contiguous(), self.init_c[i].expand(-1, bsz, -1).contiguous()

	def forward(self, input, input_lengths=None):
		bsz, slen = input.size(0), input.size(1)
		output = input
		outputs = []
		if input_lengths is not None:
			lens = input_lengths.data.cpu().numpy()

		for i in range(self.nlayers):
			hidden, c = self.get_init(bsz, i)

			output = self.dropout(output)
			if input_lengths is not None:
				output = rnn.pack_padded_sequence(output, lens, batch_first=True)

			output, hidden = self.rnns[i](output, (hidden, c))


			if input_lengths is not None:
				output, _ = rnn.pad_packed_sequence(output, batch_first=True)
				if output.size(1) < slen: # used for parallel
					padding = Variable(output.data.new(1, 1, 1).zero_())
					output = torch.cat([output, padding.expand(output.size(0), slen-output.size(1), output.size(2))], dim=1)
			if self.return_last:
				outputs.append(hidden.permute(1, 0, 2).contiguous().view(bsz, -1))
			else:
				outputs.append(output)
		if self.concat:
			return torch.cat(outputs, dim=2)
		return outputs[-1]


class SelfAttention(nn.Module):
	def __init__(self, input_size, dropout):
		super().__init__()
		# self.dropout = LockedDropout(dropout)
		self.input_linear = nn.Linear(input_size, 1, bias=False)
		self.dot_scale = nn.Parameter(torch.Tensor(input_size).uniform_(1.0 / (input_size ** 0.5)))

	def forward(self, input, memory, mask):

		# input = self.dropout(input)
		# memory = self.dropout(memory)

		input_dot = self.input_linear(input)
		cross_dot = torch.bmm(input * self.dot_scale, memory.permute(0, 2, 1).contiguous())
		att = input_dot + cross_dot
		att = att - 1e30 * (1 - mask[:,None])

		weight_one = F.softmax(att, dim=-1)
		output_one = torch.bmm(weight_one, memory)

		return torch.cat([input, output_one], dim=-1)


class BiAttention(nn.Module):
	def __init__(self, input_size, dropout):
		super().__init__()
		self.dropout = LockedDropout(dropout)
		self.input_linear = nn.Linear(input_size, 1, bias=False)
		self.memory_linear = nn.Linear(input_size, 1, bias=False)

		self.dot_scale = nn.Parameter(torch.Tensor(input_size).uniform_(1.0 / (input_size ** 0.5)))

	def forward(self, input, memory, mask):
		bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)

		input = self.dropout(input)
		memory = self.dropout(memory)

		input_dot = self.input_linear(input)
		memory_dot = self.memory_linear(memory).view(bsz, 1, memory_len)
		cross_dot = torch.bmm(input * self.dot_scale, memory.permute(0, 2, 1).contiguous())
		att = input_dot + memory_dot + cross_dot
		att = att - 1e30 * (1 - mask[:,None])

		weight_one = F.softmax(att, dim=-1)
		output_one = torch.bmm(weight_one, memory)
		weight_two = F.softmax(att.max(dim=-1)[0], dim=-1).view(bsz, 1, input_len)
		output_two = torch.bmm(weight_two, input)

		return torch.cat([input, output_one, input*output_one, output_two*output_one], dim=-1)
