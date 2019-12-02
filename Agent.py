
import torch
from Environment import env
import random
from utils import one_hot,logger,global_device
import pickle
import logging

class agent(object):
	def __init__(self,Policy,environment,policymethod='MLP',discount=1.5,reward=1):
		self.policymethod=policymethod
		self.env=environment
		self.policy=Policy
		self.discount=discount
		self.reward=reward

	def trajectory(self,e1,target_e,max_len=4):
		Observ=self.env.get_initial_state(e1,target_e)#[cur_e,cur_r,next_e,target_e]
		traj=[list(Observ)]
		success=False
		for i in range(max_len):
			cur_e,cur_r,next_e,target_e=Observ[0],Observ[1],Observ[2],Observ[3]
			logger.MARK('Observing:%s'%str(Observ))
			if next_e==target_e:
				success=True
				traj=self.get_reward(traj,success=success)
				break
			elif next_e!=target_e and i==max_len-1:
				traj.append([next_e,self.env.relid2name['OP'],self.env.entid2name['OP'],target_e])
				traj=self.get_reward(traj,success=success)
				break
			elif cur_r==self.env.relOPid:
				traj = self.get_reward(traj, success=success)
				break
			else:
				next_e, next_r, next_next_e=self.move(Observ)
				Observ=[next_e, next_r, next_next_e,target_e]
				traj.append(Observ)
		assert len(traj)>0
		return traj,success
	def get_reward(self,traj,success=False):
		if success:
			traj_reward=[i+[self.reward*self.discount**k] for k,i in enumerate(traj)]
		else:
			traj_reward=[i+[0] for k,i in enumerate(traj)]
		return traj_reward
	def move(self,state):
	
		# [cur_e,cur_r,next_e,target_e]
		cur_e,cur_r,next_e,target_e=state[0],state[1],state[2],state[3]
		if self.env.entid2name[cur_e]=='START':
			step=0
		else:
			step=1
		action_space=self.env.get_action_space(next_e)
		logger.MARK('moving>>>>>action_space:%s'%str(action_space))
		if len(action_space)>0:

			state_vec=self.env.get_state_vec(cur_e,cur_r,target_e)
			action_pro=self.move_predict(state_vec,mode=self.policymethod,step=step)
			
			action_space_pro=action_pro[action_space]
			action=torch.argmax(action_space_pro)
			
			action=action_space[action.cpu()]
			
			logger.MARK('moving>>>>>move action:%s'%str(self.env.relid2name[action]))
			if action==self.env.relOPid:
				logger.MARK('moving>>>>>moving to OP..')
				return next_e, self.env.relOPid, self.env.entOPid
			next_r=action
			next_next_e=self.env.choose_e_from_action(next_e,next_r)
			
			return next_e,next_r,next_next_e
		else:
			logger.MARK('moving>>>>>moving to OP..')
			return next_e,self.env.relOPid,self.env.entOPid
	def move_predict(self,state_vec,mode='MLP',step=None):
		if mode=='MLP':
			return self.policy.predict(state_vec)
		elif mode=='GRU':
			if step==0:
				self.policy.zero_history()
			return self.policy.predict(state_vec)

	def parameters(self):
		assert self.policy
		return list(self.policy.parameters())
	def update_policy(self,state):
		cur_e, cur_r, next_e, target_e,reward = tuple(state)
		y = one_hot(cur_r, num_class=self.env.rel_num)
		y = torch.tensor(y,dtype=torch.float32,device=torch.device(global_device))*reward

		state_vec = self.env.get_state_vec(cur_e, cur_r, target_e)
		res=self.policy.forward(state_vec)
		res=-torch.log(res)
		res=y*res
		loss=torch.sum(res)
		return loss
	def update_memory_policy(self,state):
		cur_e, cur_r, next_e, target_e, reward = tuple(state)
		y = one_hot(cur_r, num_class=self.env.rel_num)
		y = torch.tensor(y, dtype=torch.float32,device=torch.device(global_device))*reward
		res = self.policy.forward(cur_e, cur_r, target_e)
		res = -torch.log(res)*y
		loss = torch.sum(res)
		return loss

