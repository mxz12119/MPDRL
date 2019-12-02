import torch.nn as nn
import torch
from torch.nn.init import xavier_normal_, xavier_uniform_
import torch.nn.functional as F
from utils import global_device
from utils import logger
import random
import numpy as np
class gate(nn.Module):
	def __init__(self,input,output,activation='sigmoid'):
		super(gate, self).__init__()
		self.l_x=nn.Linear(input,output,bias=False)
		self.l_h=nn.Linear(output,output,bias=False)
		xavier_uniform_(self.l_x.weight.data)
		xavier_uniform_(self.l_h.weight.data)
		if activation=='sigmoid':
			self.activation=torch.sigmoid
		elif activation=='tanh':

			self.activation=torch.tanh

		elif activation=='relu':
			self.activation=torch.nn.functional.relu
	def forward(self, x,h):
		res1=self.l_x(x)
		res2=self.l_h(h)

		res=self.activation(res1+res2)
		return res

class Policy_memory(nn.Module):
	def __init__(self,env,input=300,hidden=100,output=33):
		super(Policy_memory,self).__init__()
		self.hidden=hidden
		self.resgate=gate(input,hidden,'sigmoid')
		self.zgate=gate(input,hidden,'sigmoid')

		self.enco=gate(input,hidden,'tanh')
		self.policy_decoder=nn.Linear(hidden+input,output,bias=True)
		xavier_uniform_(self.policy_decoder.weight.data)
		#self.register_parameter('hidden_vec',nn.Parameter(torch.zeros(hidden),requires_grad=False))
		self.emb=nn.Embedding(40,300)
		self.history=[]
		self.env=env
	def zero_history(self):
		self.history=[]
	def forward(self, ent,rel,target_e):

		hidden_vec=torch.zeros(self.hidden,device=torch.device(global_device))
		if len(self.history)>0:
			for i in self.history:
				e_h,r_h,target_e_h=i[0],i[1],i[2]
				tt=self.env.get_state_vec(e_h,r_h,target_e_h)

				r = self.resgate(tt, hidden_vec)
				z = self.zgate(tt, hidden_vec)
				hidden_layer = self.enco(tt, hidden_vec * r)
				hidden_vec = z * hidden_layer + (1 - z) + hidden_vec

		input=self.env.get_state_vec(ent,rel,target_e)
		r = self.resgate(input, hidden_vec)
		z = self.zgate(input, hidden_vec)
		hidden_layer = self.enco(input, hidden_vec * r)
		next_hidden = z * hidden_layer + (1 - z) + hidden_vec
		cat = torch.cat((input, next_hidden))
		y = self.policy_decoder(cat)
		y = F.relu(y)
		y = F.softmax(y)
		self.history.append([ent,rel,target_e])
		#logger.debug('self.memory.l_h.weight.data.grad'+str(self.memory.l_h.weight.data.grad))
		return y
	def predict(self,input):
		with torch.no_grad():
			#cpu->cuda
			hidden_vec = torch.zeros(self.hidden,device=torch.device(global_device))
			if len(self.history) > 0:
				for i in self.history:
					r = self.resgate(i, hidden_vec)
					z = self.zgate(i, hidden_vec)
					hidden_layer = self.enco(i, hidden_vec * r)
					hidden_vec = z * hidden_layer + (1 - z) + hidden_vec
			r = self.resgate(input, hidden_vec)
			z = self.zgate(input, hidden_vec)
			hidden_layer = self.enco(input, hidden_vec * r)
			next_hidden = z * hidden_layer + (1 - z) + hidden_vec
			cat = torch.cat((input, next_hidden))
			y = self.policy_decoder(cat)
			y = F.relu(y)
			y = F.softmax(y)
			self.history.append(input)
			# logger.debug('self.memory.l_h.weight.data.grad'+str(self.memory.l_h.weight.data.grad))
			return y


