import random
from Environment import env
import numpy as np
from sklearn.linear_model import LinearRegression,LogisticRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn import feature_selection
from itertools import product
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import json
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor




def get_feature_set(pathfile,mode='relpath'):
	with open(pathfile,'r') as fin:
		Feature_SET={}
		for i in fin.readlines():
			line=i.strip().split()
			if len(line)%2==0:
				line=line[:-1]
			feature=[]
			if mode=='relpath':
				for k,i in enumerate(line):
					if k%2==1:
						feature.append(i)
			elif mode=='metapath':
				feature=line
			feature=tuple(feature)

			if feature in Feature_SET:
				Feature_SET[feature]+=1
			else:
				Feature_SET[feature]=1
	return Feature_SET
class linkprediction:
	def __init__(self,task_relation='<isCitizenOf>',rootpath = './data/yagosmall/'):
		self.rootpath=rootpath+task_relation+'/'
		self.datapath = {'type2id': self.rootpath + 'type2id.json', 'relation2id': self.rootpath + 'relation2id.json', \
					'graph': self.rootpath + 'graph.pkl', 'ent2type': self.rootpath + 'ent2type.json', \
					'entity2id': self.rootpath + 'entity2id.json',}

		self.init_env(task_relation)
		self.task_relation=task_relation
		self.get_LR_dataset()
		self.get_training_exples()
		self.LR_result={}


	def init_env(self,task_relation):
		self.Env_a = env(self.datapath)
		self.Env_a.init_relation_query_state(task_relation)


	def get_LR_dataset(self,test_pairs='test_positive_pairs'):
		with open(self.rootpath+self.task_relation+test_pairs,'r') as fin:
			self.pairs=[]
			for i in fin.readlines():
				line=i.strip().split()
				self.pairs.append([line[0],line[1]])
		self.typelist_e1={}
		self.typelist_e2=set([])
		self.query=[(i[0],i[1]) for i in self.Env_a.query]
		for i in self.pairs:
			e1,e2=i[0],i[1]
			e1,e2=self.Env_a.entity2id[e1],self.Env_a.entity2id[e2]
			for j in self.Env_a.ent2type_id[e1]:
				s=self.Env_a.typeid2name[j]
				if s in self.typelist_e1:
					self.typelist_e1[s]+=1
				else:
					self.typelist_e1[s]=1
			for m in self.Env_a.ent2type_id[e2]:
				self.typelist_e2.add(self.Env_a.typeid2name[m])
		print('len(typelist_e1):',len(self.typelist_e1))
		print('len(typelist_e2):',len(self.typelist_e2))
		sort_dict=sorted(self.typelist_e1.items(),key=lambda x:x[1],reverse=True)
		self.typelist_e1=set([i[0] for i in sort_dict])
	def cls_balence(self,pos,neg,ran=(0.3,3)):
		ro = len(pos) / len(neg)
		down,up=0.3,3
		if ro > up:
			keep = len(neg) * up
			random.shuffle(pos)
			pos=pos[:keep]
		elif ro < down:
			keep = int(len(neg) / down)
			random.shuffle(neg)
			neg=neg[:keep]
		return pos,neg

	def get_training_exples(self):
		pairs = [tuple(i) for i in self.pairs]
		neg_pairs = []
		rn = list(range(len(self.Env_a.triple)))
		random.shuffle(rn)
		for k in rn:
			t = self.Env_a.triple[k]
			e1, e2 = t[0], t[1]
			neg_pair_p = self.Env_a.near_negative_sampling(e1, self.typelist_e2)
			if len(neg_pair_p)>0:
				for neg_pair in neg_pair_p:
					# if neg_pair is not None and neg_pair not in query and sum(get_feature(neg_pair, Env_a)) >= 1:
					if (neg_pair is not None) and (neg_pair not in self.query):
						neg_pairs.append(neg_pair)
					if len(neg_pairs) % 100 == 0 and len(neg_pairs) > 0:
						print('len(neg pairs):', len(neg_pairs))
			if len(neg_pairs) >= int(0.5*len(pairs)):
				break
		len_exa = min([len(pairs), len(neg_pairs)])

		print('found %d suitable positive pair and %d suitable negative pair'%(len(pairs),len(neg_pairs)))
		random.shuffle(pairs)
		#pairs,neg_pairs=self.cls_balence(pairs,neg_pairs)
		all_pairs = pairs + neg_pairs
		y_pairs = [1] * len(pairs) + [0] * len(neg_pairs)
		print('sample number:',len(all_pairs))
		print('pos sample:',len(pairs))
		print('neg sample:',len(neg_pairs))
		rn = list(range(len(all_pairs)))
		random.shuffle(rn)
		with open(self.rootpath + 'pairs_examples', 'w') as fin:
			wst = []
			for i in rn:
				wst.append(all_pairs[i][0] + '\t' + all_pairs[i][1] + '\t' + str(y_pairs[i]))
			wst = '\n'.join(wst)

			fin.write(wst)
		self.pairs_examples = [all_pairs[i] for i in rn]
		self.pairs_examples_y = [y_pairs[i] for i in rn]
		return self.pairs_examples, self.pairs_examples_y
	def learner(self,name=None):
		if name=='cos':
			pass


	def __call__(self,pathfile,exp_name=None,mode='metapath'):
		Feature_set=get_feature_set(self.rootpath+pathfile,mode)
		print('Feature dimention:',len(Feature_set))

		print(list(Feature_set.keys())[:10])

		pairs_feature = []
		for k, i in enumerate(self.pairs_examples):
			#print('test pair:',i,self.pairs_examples_y[k])
			t = get_feature(i, self.Env_a,Feature_set,mode)
			pairs_feature.append(t)

		pairs_feature = np.array(pairs_feature)
		train = pairs_feature[:int(len(self.pairs_examples) * 0.6)]
		test = pairs_feature[int(len(self.pairs_examples) * 0.6):]
		y = np.array(self.pairs_examples_y)
		train_y = y[:int(len(self.pairs_examples) * 0.6)]
		test_y = y[int(len(self.pairs_examples) * 0.6):]
		model=Lasso(alpha=0.00001)

		model.fit(train, train_y)
		pre_y = model.predict(test)
		print('pre_y:',pre_y)

		Feature_set_list = list(Feature_set.keys())
		coef=list(model.coef_)
		srt_coef=sorted(zip(coef,list(range(len(coef)))),key=lambda x:x[0],reverse=True)
		print('LASSO top 10 coef:')
		for i in range(10):
			print('coef:',srt_coef[i][0])
			t = srt_coef[i][1]
			print(Feature_set_list[t])
		fpr, tpr, thresholds = roc_curve(test_y, pre_y)

		AUC = auc(fpr, tpr)
		scoreFun = feature_selection.chi2
		bestK=min(5,pairs_feature.shape[1])
		sele = feature_selection.SelectKBest(score_func=scoreFun, k=bestK)
		path_evaluation=sele.fit(pairs_feature,y)
		path_scores=list(path_evaluation.scores_)
		path_scores=[0 if np.isnan(i) else i for i in path_scores]
		sort_path_scores=sorted(zip(path_scores,list(range(len(path_scores)))),key=lambda x:x[0],reverse=True)

		print('top 10 path scores:')
		for i in range(10):
			print(sort_path_scores[i][0])
			t=sort_path_scores[i][1]
			print(Feature_set_list[t])

		print('pvalues:',list(sele.pvalues_))
		print('AUC:',AUC)
		expp_res={'AUC':AUC,'fpr':fpr,'tpr':tpr,'model':model,'Feature':list(Feature_set.keys()),'path_weights':list(path_evaluation.scores_),'pvalues':list(sele.pvalues_)}
		if exp_name==None:
			exp_name=pathfile
		self.LR_result[exp_name]=expp_res
	def draw(self):
		if len(self.LR_result)>0:
			print('-------------------Report-------------------------')
			for exp in self.LR_result:
				plt.plot(self.LR_result[exp]['fpr'],self.LR_result[exp]['tpr'],label=exp)
				print('AUC(%s):%.3f'%(exp,self.LR_result[exp]['AUC']))
			print('--------------------------------------------------')
			plt.xlabel('False Positive Rate')
			plt.ylabel('True Positive Rate')
			plt.legend()
			figname=self.rootpath+self.task_relation+'_link_prediction.png'
			plt.savefig(figname)
			self.write_result(figname)
			plt.close('all')

		else:
			print('Exp empty!')
	def write_result(self,file):
		wtr={}
		for i in self.LR_result:
			wtr[i]={}
			wtr[i]['fpr']=list(self.LR_result[i]['fpr'])
			wtr[i]['tpr'] = list(self.LR_result[i]['tpr'])
			wtr[i]['AUC']=self.LR_result[i]['AUC']
		with open(file+'.json','w') as fin:
			json.dump(wtr,fin)

def get_feature(pair,Env,Feature_SET,mode='relpath'):
	e1,e2=pair[0],pair[1]
	paths = Env.path_traverse_BFS(e1, e2, max_len=4)
	rel_path = []
	ent_path=[]
	for i in paths:
		L = []
		M=[]
		for k,j in enumerate(i):
			if k%2==0:

				M.append(j)
			else:
				L.append(j)
		rel_path.append(tuple(L))
		ent_path.append(tuple(M))

	feature_vecotr = {i: 0 for i in Feature_SET.keys()}
	if mode=='relpath':
		for i in rel_path:
			if i in feature_vecotr:
				feature_vecotr[i] = 1
	elif mode=='metapath':
		metapath = []
		for ent_p,rel_p in zip(ent_path,rel_path):
			typeinfo=[Env.ent2type[kk] for kk in ent_p]
			path_l = len(ent_p)+len(rel_p)
			metapath_u = [0] * path_l
			for k, r in enumerate(rel_p):
				metapath_u[k * 2 + 1] = r
			for ty_u in product(*typeinfo):
				for k, t in enumerate(ty_u):
					metapath_u[k * 2] = t
				try:
					metapath.append(tuple(metapath_u))
				except:
					print('generate metapth feature error:',tuple(metapath_u))
		for i in metapath:
			if i in feature_vecotr:
				feature_vecotr[i] = 1
	res=list(feature_vecotr.values())

	return res

def test_nell():

	rootpath = './data/NELL/'
	task_relation_set = ['concept:teamplaysagainstteam',
						'concept:competeswith',
						 'concept:worksfor']
	for task_relation in task_relation_set:
			path = [task_relation + '.metapathswalk']
			task = linkprediction(task_relation,rootpath)
			for p in path:
				task(p, mode='metapath')
			task.draw()
def test_yago():
	rootpath = './data/yagosmall/'
	task_relation = "<wasBornIn>"

	path = [task_relation + ".metapaths"]
	task = linkprediction(task_relation)
	for p in path:
		task(p, mode='relpath')
	task.draw()
if __name__=='__main__':
	test_nell()

