import torch
import random
import json
import pickle
from queue import Queue
import matplotlib.pyplot as plt
import logging
from utils import normalize,logger,global_device
from torch.nn.init import xavier_normal_, xavier_uniform_
class node:
	def __init__(self,x,y=None,pre=None,next=None,level=1):
		self.x=x
		self.y=y
		self.pre=pre
		self.next=next
		self.level=level
	def back(self):
		M=[(self.y,self.x)]
		t = self.pre
		while t!=None:
			M.append((t.y,t.x))
			t=t.pre
		return M[::-1]

class env(object):
	def __init__(self,datapath,pretrain=None,dim=100):
		self.vec_dim=dim
		for k,v in datapath.items():
			setattr(self,k,self.load_file(v))
			logger.info('load env.%s success...'%k)
		self.triple_num = len(self.graph)
		
		self.init_data()
		if pretrain:
			self.init_embedding_from_pretrain()
		else:
			self.init_embedding()

		
		self.triple2adj()
		self.parameter={}
	def init_data(self):
		assert len(self.relation2id) not in self.relation2id.values()

		self.type2id['OP'] = len(self.type2id)
		self.type2id['START'] = len(self.type2id)
		self.ent2type['OP'] = ['OP']
		self.ent2type['START'] = ['START']
		self.entity2id['OP'] = len(self.entity2id)
		self.entity2id['START'] = len(self.entity2id)

		self.ent2type_id = {}
		for k, v in self.ent2type.items():
			self.ent2type_id[self.entity2id[k]] = [self.type2id[i] for i in v]

		self.entid2name = {v: k for k, v in self.entity2id.items()}
		self.typeid2name = {v: k for k, v in self.type2id.items()}
		self.relation2id['OP'] = len(self.relation2id)
		self.relation2id['START'] = len(self.relation2id)
		self.relid2name = {v: k for k, v in self.relation2id.items()}

		self.type_num = len(self.type2id)
		self.ent_num = len(self.entity2id)

		self.rel_num = len(self.relation2id)

		self.relOPid = self.relation2id['OP']
		self.relStartid = self.relation2id['START']
		self.entOPid = self.entity2id['OP']
		self.entStartid = self.entity2id['START']

	def init_embedding(self):


		self.rel_vec = torch.nn.Embedding(self.rel_num, self.vec_dim)
		xavier_uniform_(self.rel_vec.weight.data)
		self.type_vec = torch.nn.Embedding(self.ent_num, self.vec_dim)
		xavier_uniform_(self.type_vec.weight.data)
		if global_device=='cuda:0':
			self.rel_vec=self.rel_vec.cuda()
			self.type_vec=self.type_vec.cuda()


	def init_embedding_from_pretrain(self):
		self.rel_vec = self.embeddings['rel_embeddings'].copy()
		self.vec_dim = len(self.rel_vec[0])
		relOP_vec = [(random.random() - 0.5) / self.vec_dim ** 0.5 for _ in range(self.vec_dim)]
		relOP_vec = normalize(relOP_vec, norm=2)
		relStart_vec = [(random.random() - 0.5) / self.vec_dim ** 0.5 for _ in range(self.vec_dim)]
		relStart_vec = normalize(relStart_vec, norm=2)

		self.rel_vec.append(relOP_vec)
		self.rel_vec.append(relStart_vec)


		self.type_vec = self.embeddings['ent_embeddings'].copy()
		entOP_vec = [(random.random() - 0.5) / self.vec_dim ** 0.5 for _ in range(self.vec_dim)]
		entOP_vec=normalize(entOP_vec,norm=2)

		entStart_vec = [(random.random() - 0.5) / self.vec_dim ** 0.5 for _ in range(self.vec_dim)]
		entStart_vec=normalize(entStart_vec,norm=2)
		entOP_vec = torch.tensor(entOP_vec, dtype=torch.float32)
		entStart_vec = torch.tensor(entStart_vec, dtype=torch.float32)

		self.type_vec.append(entOP_vec)

		self.type_vec.append(entStart_vec)


		del self.embeddings

		rel_vec = torch.nn.Embedding(self.rel_num, self.vec_dim)
		rel_vec.from_pretrained(torch.tensor(self.rel_vec), freeze=False)
		self.rel_vec = rel_vec

		type_vec = torch.nn.Embedding(self.ent_num, self.vec_dim)
		type_vec.from_pretrained(torch.tensor(self.type_vec), freeze=False)
		self.type_vec = type_vec
		if global_device=='cuda:0':
			self.rel_vec=self.rel_vec.cuda()
			self.type_vec=self.type_vec.cuda()
		print('typevec device:',self.type_vec.weight.device)

	def parameters(self):
		return list(self.rel_vec.parameters())+list(self.type_vec.parameters())
		
	def init_relation_query_state(self,relation):
		assert self.graph_state == 'adj'
		if type(relation)==str:
			relation=self.relation2id[relation]
		kb = {}
		kb_inv = {}
		self.query=[]
		for i in self.triple:
			e1, e2, r = i[0], i[1], i[2]
			e1, e2, r = self.entity2id[e1], self.entity2id[e2], self.relation2id[r]
			if relation==r:
				self.query.append(i)
			else:
				if e1 in kb:
					kb[e1].append((r, e2))
				else:
					kb[e1] = [(r, e2)]
				if e2 in kb_inv:
					kb_inv[e2].append((r, e1))
				else:
					kb_inv[e2] = [(r, e1)]
		self.graph = kb
		self.graph_inv = kb_inv
		return self.query
			
			
			
	def triple2adj(self):
		
		assert hasattr(self,'graph')
		kb={}

		for i in self.graph:
			e1,e2,r=i[0],i[1],i[2]
			e1,e2,r=self.entity2id[e1],self.entity2id[e2],self.relation2id[r]
			if e1 in kb:
				kb[e1].append((r,e2))
			else:
				kb[e1]=[(r,e2)]

		self.triple=self.graph.copy()
		self.graph=kb
		self.graph_state='adj'
		
		
		
	def load_file(self,path):
		if path[-5:]=='.json':
			with open(path,'r') as fin:
				file=json.load(fin)
		elif path[-4:]=='.pkl':
			with open(path,'rb') as fin:
				file=pickle.load(fin)
		else:
			with open(path,'r') as fin:
				file=[]
				for i in fin.readlines():
					file.append(i.stripl().split())
		return file
		
	def get_initial_state(self,e1,target_e):
		next_e=e1
		cur_r,cur_e=self.relStartid,self.entStartid
		return cur_e,cur_r,next_e,target_e
	def get_neighbor_relation(self,ent):
		assert self.graph_state=='adj'
		path_list=self.graph[ent]
		neighbor_relation=[i[0] for i in path_list]
		
		assert len(neighbor_relation)>0
		return neighbor_relation
	def get_action_space(self,ent):
		assert self.graph_state == 'adj'
		try:
			path_list = self.graph[ent]
		except:
			logger.debug('Key error:%s'%str(self.entid2name[ent]))
			return []
		
		res=set([])
		for i in path_list:
			res.add(i[0])
		res.add(self.relation2id['OP'])
		
		assert len(res) > 0
		return list(res)
		
		
	def get_state_vec(self,cur_e,cur_r,target_e,mode='cat'):
		if mode=='cat':
			cur_e_vec=self.entid2vec(cur_e)
			cur_r_vec=self.relid2vec(cur_r)
			tar_e_vec=self.entid2vec(target_e)
			return torch.cat((cur_e_vec,cur_r_vec,tar_e_vec),dim=0)
	def entid2vec(self,ent,max_type=30):

		type_list=self.ent2type_id[ent]
		type_list=torch.tensor(type_list,device=torch.device(global_device))
		ent_type_sum=self.type_vec(type_list)
		ent_type_sum=torch.sum(ent_type_sum,dim=0)
		return ent_type_sum
	def relid2vec(self,rel):#int

		rel=torch.tensor(rel,device=torch.device(global_device))
		return self.rel_vec(rel)
	def choose_e_from_action(self,ent,action):
		path_list=self.graph[ent]
		L=[]
		for i in path_list:
			if i[0]==action:
				L.append(i[1])
		return random.choice(L)
	def traj2list(self,traj):
		L=[]
		for i in traj[1:]:
			e1,r=i[0],i[1]
			if e1 != 'OP' and e1 != 'START':
				e1=self.entid2name[e1]
			if r!='OP' and r!='START':
				r=self.relid2name[r]
			L.append(e1)
			L.append(r)
		L.append(self.entid2name[traj[-1][2]])
		return L

	def traj_for_showing(self,traj):
		res_str=[]
		for i in traj:
			e1,r,e2=i[0],i[1],i[2]
			if e1 != 'OP' and e1 != 'START':
				e1=self.entid2name[e1]
			if e2!='OP' and e2!='START':
				e2=self.entid2name[e2]
			if r!='OP' and r!='START':
				r=self.relid2name[r]
			
			fil=e1+'~~'+r+'~~'+e2
			res_str.append(fil)
		res_str='==='.join(res_str)
		
		res_str=res_str.replace('>','')
		res_str=res_str.replace('<','')
		return res_str

	def In_type(self,e, typelist):
		if type(e) == int:
			e = self.entid2name[e]
		typelist_e = self.ent2type[e]
		for m in typelist_e:
			if m in typelist:
				return True

		return False
	def near_negative_sampling(self,ent,target_type,max_len=7,lowest_level=1):
		if type(ent)==str:
			ent=self.entity2id[ent]
		q = Queue()
		q.put(node(ent, level=1))

		l = 1

		negtive_paris = []

		while not q.empty():
			cur = q.get()

			if cur.level>=lowest_level and self.In_type(cur.x,target_type):
				negtive_paris.append((ent,cur.x))
			if cur.x in self.graph and cur.level <= max_len:
				if len(self.graph[cur.x]) > 0:
					l += 1
					for i in self.graph[cur.x]:
						q.put(node(i[1], pre=cur, level=l, y=i[0]))
		if len(negtive_paris)>=3:
			random.shuffle(negtive_paris)
			return [(self.entid2name[t[0]], self.entid2name[t[1]]) for t in negtive_paris[:3]]
		elif len(negtive_paris)>0:

			return [(self.entid2name[t[0]], self.entid2name[t[1]]) for t in negtive_paris]
		else:
			return []
	def iscircle(self,path):
		if len(set(path))==len(path):
			return False
		else:
			return True
	def path_traverse_BFS(self,e1,e2,max_len=4):
		if type(e1)==str:
			e1=self.entity2id[e1]
		if type(e2)==str:
			e2=self.entity2id[e2]
		q = Queue()
		q.put(node(e1,level=1))

		l = 1

		paths=[]
		while not q.empty():
			cur = q.get()

			if cur.x == e2:
				backpath=cur.back()
				if not self.iscircle(backpath):
					paths.append(backpath)
			if cur.x in self.graph and cur.level <= max_len:
				if len(self.graph[cur.x]) > 0:
					l += 1
					for i in self.graph[cur.x]:

						q.put(node(i[1],pre=cur,level=l,y=i[0]))


		M=[]
		for i in paths:
			L=[]
			for j in i:
				if j[0] is not None:
					L.append(self.relid2name[j[0]])
				L.append(self.entid2name[j[1]])
			M.append(L)
		return M
	def BFS(self,e1,e2,max_len=4):
		q=Queue()
		q.put(e1)
		mark_q=Queue()
		l=1
		mark_q.put(l)
		while not q.empty():
			cur=q.get()
			cur_l=mark_q.get()
			if cur==e2:
				return True
			if cur in self.graph and l<=max_len:
				if len(self.graph[cur])>0:
					l+=1
					for i in self.graph[cur]:
						q.put(i[1])
						mark_q.put(l)
		return False
	def filter_query(self,max_len=5,maxnum=None):
		assert len(self.query)>0
		self.filter_query=[]
		count=0
		if maxnum is None:
			maxnum=10000000
		for k, fact in enumerate(self.query):
			e1, e2 = fact[0], fact[1]
			e1, e2 = self.entity2id[e1], self.entity2id[e2]
			res = self.BFS(e1, e2, max_len=max_len)
			if res:
				count+=1
				self.filter_query.append(fact)
			if k % int(len(self.query) * 0.1) == 0 and k > 0:
				logger.info("%%%.3f test..." % (k * 100 / len(self.query)))
			if count>=maxnum:
				logger.info('Got %d queries..'%count)
				break
		logger.info("%%%.3f have been left.." % (100*(count / len(self.query))))

def txt2json(path):
	res={}
	with open(path,'r') as fin:
		for i in fin.readlines():
			line=i.strip().split()
			if len(line)==2:
				res[line[0]]=int(line[1])
	if path[-4:]=='.txt':
		path=path[:-4]+'.json'
		with open(path,'w') as fin:
			json.dump(res,fin)
	return res







	
	
	
	
		
		
	
	