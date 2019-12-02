import torch

import torch
from Environment import env
from Functions import Policy_memory
import random
from Agent import agent
import json
from utils import one_hot
import pickle
import logging
from utils import logger,global_device
import os
from itertools import product
from typetree import typetree
random.seed(141)
torch.manual_seed(77)
def preprocessing(facts,ent2type,savefolder=None):
    if isinstance(facts,str):
        with open(facts,'r') as fin:
            res=[]
            for i in fin.readlines():
                res.append(i.strip().split())
        facts=res
    if isinstance(ent2type,str):
        with open(ent2type,'r') as fin:
            ent2type=json.load(fin)
    ent=set([])
    rel=set([])
    for i in facts:
        e1,r,e2=tuple(i)
        ent.add(e1)
        ent.add(e2)
        rel.add(r)
    assert len(ent)==len(ent2type)
    ty=set([])
    for typelist in ent2type.values():
        for t in typelist:
            ty.add(t)
    if not os.path.exists(savefolder):
        os.mkdir(savefolder)

    with open(savefolder+'/entity2id.json','w') as fin:
        entity2id={v:k for k,v in enumerate(ent)}
        json.dump(entity2id,fin)
    with open(savefolder+'/relation2id.json','w') as fin:
        relation2id={v:k for k,v in enumerate(rel)}
        json.dump(relation2id,fin)
    with open(savefolder+'/type2id.json','w') as fin:
        type2id={v:k for k,v in enumerate(ty)}
        json.dump(type2id,fin)
    with open(savefolder+'/ent2type.json','w') as fin:
        json.dump(ent2type,fin)
    with open(savefolder+'/graph.pkl','wb') as fin:
        facts=[[i[0],i[2],i[1]] for i in facts]
        pickle.dump(facts,fin)


def train(task_relation="<diedIn>",rootpath=None,epoch=5):
    datapath = {'type2id': rootpath + 'type2id.json', 'relation2id': rootpath + 'relation2id.json', \
         'graph': rootpath + 'graph.pkl', 'ent2type': rootpath + 'ent2type.json' ,\
         'entity2id': rootpath + 'entity2id.json'}
    Env_a = env(datapath)
    Env_a.init_relation_query_state(task_relation)
    batchsize=20
    maxlen=5
    po = Policy_memory(Env_a,300, 100, Env_a.rel_num)
    Env_a.filter_query(maxlen,5000)
    pairs = Env_a.filter_query
    random.shuffle(pairs)

    #training_pairs=pairs[int(len(pairs)*0.9):]
    training_pairs=pairs
    test_pairs=pairs[:int(len(pairs)*0.5)]
    # with open(rootpath+'test_positive_pairs','w') as fin:
    #     wstr=[]
    #     for i in test_pairs:
    #         wstr.append(str(i[0]+'\t'+str(i[1])))
    #     wstr='\n'.join(wstr)
    #     fin.write(wstr)
    # return
    reward_record=[]
    success_record=[]
    path_length=0
    valid_paris=pairs[int(len(pairs)*0.5):int(len(pairs)*0.6)]
    print('Train pairs:',len(training_pairs))
    print('valid pairs:',len(valid_paris))
    print('Test pairs:',len(test_pairs))
    agent_a = agent(po, Env_a,policymethod='GRU')
    if global_device=='cuda:0':
        po=po.cuda()

    try_count, batch_loss, ave_reward, ave_success = 0, 0, 0, 0
    opt=torch.optim.Adam(agent_a.parameters()+Env_a.parameters(),lr=0.001)
    for ep in range(epoch):
        opt.zero_grad()
        random.shuffle(training_pairs)
        for query in training_pairs:
            try:
                e1, e2 = query[0], query[1]
                e1, e2 = Env_a.entity2id[e1], Env_a.entity2id[e2]
                with torch.no_grad():
                    traj, success = agent_a.trajectory(e1, e2,max_len=maxlen)
                try_count += 1
            except KeyError:
                continue




            logger.MARK(Env_a.traj_for_showing(traj))

            traj_loss=0
            po.zero_history()
            traj_reward=0

            for i in traj:

                ave_reward+=i[4]
                traj_reward+=i[4]
                loss=agent_a.update_memory_policy(i)
                loss.backward()
                traj_loss+=loss.cpu()
            if success:
                ave_success+=1
                path_length += len(traj) - 1
                success_record.append(1)
            else:
                success_record.append(0)
            reward_record.append(traj_reward)
            batch_loss+=traj_loss/len(traj)
            #logger.info(str(po.memory.l_h.weight))
            #logger.info(str(po.enco.l_h.weight))
            if try_count%batchsize==0 and try_count>0:
                opt.step()
                opt.zero_grad()
                logger.info('|%d epoch|%d eposide|Batch_loss:%.4f|Ave_reward:%.3f|Ave_success:%%%.2f|ave path lenghth:%.2f|'%(ep,try_count,batch_loss*100/batchsize,ave_reward/batchsize,ave_success*100/batchsize,path_length/ave_success))
                batch_loss,ave_reward,ave_success,path_length=0,0,0,0

            if try_count%(20*batchsize)==0 and try_count>0:
                valid(valid_paris,Env_a,agent_a,batchsize,maxlen)

        generate_paths(Env_a,agent_a,test_pairs,rootpath+task_relation+'.paths',maxlen)

    success=ave_smooth(success_record,20)
    reward=ave_smooth(reward_record,20)

    with open(rootpath+task_relation+'sucess_record_without.txt','w') as fin:
        wstr='\n'.join([str(i) for i in success])
        fin.write(wstr)
    with open(rootpath+task_relation+'reward_record_without.txt','w') as fin:
        wstr='\n'.join([str(i) for i in reward])
        fin.write(wstr)

    with open(rootpath+task_relation+'test_positive_pairs','w') as fin:
        wstr=[]
        for i in test_pairs:
            wstr.append(str(i[0]+'\t'+str(i[1])))
        wstr='\n'.join(wstr)
        fin.write(wstr)
def chunk(x,size=5):
    res=[]
    chu=[]
    for k,i in enumerate(x):
        if k==len(x):
            chu.append(i)
            res.append(chu)
        elif k%size==0 and k>0:
            res.append(chu)
            chu=[]
        chu.append(i)
    return res
def ave_smooth(x,width=20):
    x_c=chunk(x,width)
    res=[]
    for i in x_c:
        t=sum(i)/len(i)
        res.append(t)
    return res

def write_test_pairs(task_relation="<diedIn>",rootpath=None):
    datapath = {'type2id': rootpath + 'type2id.json', 'relation2id': rootpath + 'relation2id.json', \
                'graph': rootpath + 'graph.pkl', 'ent2type': rootpath + 'ent2type.json', \
                'entity2id': rootpath + 'entity2id.json'}
    Env_a = env(datapath)
    Env_a.init_relation_query_state(task_relation)

    maxlen = 8
    Env_a.filter_query(maxlen, 5000)
    pairs = Env_a.filter_query
    random.shuffle(pairs)
    test_pairs = pairs[:int(len(pairs) * 0.5)]
    with open(rootpath+'test_positive_pairs','w') as fin:
        wstr=[]
        for i in test_pairs:
            wstr.append(str(i[0]+'\t'+str(i[1])))
        wstr='\n'.join(wstr)
        fin.write(wstr)
    return


def generate_paths(Env_a,agent_a,pairs,save_path,maxlen):
    try_count, batch_loss, ave_reward, ave_success = 0, 0, 0, 0
    paths=[]
    for query in pairs:
        try:
            e1, e2 = query[0], query[1]
            e1, e2 = Env_a.entity2id[e1], Env_a.entity2id[e2]
            with torch.no_grad():
                traj, success = agent_a.trajectory(e1, e2, max_len=maxlen)
            try_count += 1
        except KeyError:
            continue

        if success:
            logger.MARK('Find paths on the test:' + Env_a.traj_for_showing(traj))
            L = Env_a.traj2list(traj)
            paths.append(L)
    with open(save_path,'w') as fin:
        wrt_str=['\t'.join(i) for i in paths]
        wrt_str='\n'.join(wrt_str)
        fin.write(wrt_str)

def LCA(ent,ent2type,ontology):
    typelist=ent2type[ent]
    LCAlist=set([])
    for i in typelist:
        stack=[]
        stack.append(i)
        while len(stack)!=0:
            t=stack.pop()
            if ontology[t]!=0:
                pass

def traj2list(traj):
    L=[]
    for i in traj[1:]:

        cur_e,cur_r,next_e=i[0],i[1],i[2]
        L.append(cur_e,cur_r)
    assert L[-1]!='OP'
    return L




def valid(pairs,enviro,ag,batchsize,maxlen=None):

    valid_pairs=pairs
    random.shuffle(valid_pairs)
    try_count, batch_loss, ave_reward, ave_success = 0, 0, 0, 0

    for query in valid_pairs:
        try:
            e1, e2 = query[0], query[1]
            e1, e2 = enviro.entity2id[e1], enviro.entity2id[e2]
            with torch.no_grad():
                traj, success = ag.trajectory(e1, e2, max_len=maxlen)
            try_count += 1
        except KeyError:
            continue

        logger.MARK(enviro.traj_for_showing(traj))

        traj_loss = 0
        with torch.no_grad():
            ag.policy.zero_history()
            for i in traj:
                ave_reward += i[4]
                loss = ag.update_memory_policy(i)
                traj_loss += loss.cpu()
        if success:
            ave_success += 1
        batch_loss += traj_loss / len(traj)


    logger.info('|%d have been valided|Batch_loss:%.4f|Ave_reward:%.3f|Ave_success:%%%.2f|' % (\
    try_count, batch_loss * 100 / try_count, ave_reward / try_count, ave_success *100 / try_count))

def PCRW(task_relation = '<isAffiliatedTo>',MAXLEN=5):
    rootpath = './data/yagosmall/' + task_relation + '/'

    datapath = {'type2id': rootpath + 'type2id.json', 'relation2id': rootpath + 'relation2id.json', \
                'graph': rootpath + 'graph.pkl', 'ent2type': rootpath + 'ent2type.json', \
                'entity2id': rootpath + 'entity2id.json'}
    Env_a = env(datapath)
    Env_a.init_relation_query_state(task_relation)
    maxlen=MAXLEN
    Env_a.filter_query(maxlen)
    pairs = Env_a.filter_query
    random.shuffle(pairs)

    training_pairs = pairs[int(len(pairs) * 0.9):]
    test_pairs = pairs[:int(len(pairs) * 0.5)]
    if len(test_pairs)>5000:
        test_pairs=test_pairs[:5000]
    allpaths=[]
    for i in test_pairs:
        e1,e2=i[0],i[1]
        paths=Env_a.path_traverse_BFS(e1,e2,max_len=MAXLEN)

        allpaths.append(paths)
    with open(rootpath+'PCRW-%d'%MAXLEN+task_relation,'w') as fin:
        wrt_str=[]
        for paths in allpaths:
            M=[]
            for path in paths:
                M.append('\t'.join(path))
            wrt_str.append('\n'.join(M))
        wrt_str='\n'.join(wrt_str)
        fin.write(wrt_str)
    with open(rootpath+'PCRW-%d'%MAXLEN+task_relation, 'r') as fin:
        Feature_SET = {}
        for i in fin.readlines():
            line = i.strip().split()
            feature = []
            for k, i in enumerate(line):
                if k % 2 == 1:
                    feature.append(i)
            feature = tuple(feature)
            if feature in Feature_SET:
                Feature_SET[feature] += 1
            else:
                Feature_SET[feature] = 1
    print('Path feature:',Feature_SET)
def load_paths(paths):
    res=[]
    with open(paths,'r') as fin:
        for i in fin.readlines():
            res.append(i.strip().split())
    ent_ser=[]
    rel_ser=[]
    for path in res:
        ent=[]
        rel=[]
        for k,i in enumerate(path):
            if k%2==0:
                ent.append(i)
            else:
                rel.append(i)
        ent_ser.append(ent)
        rel_ser.append(rel)
    return ent_ser,rel_ser
def generate_metapath(paths,ent2type,save_path,MAX_TYPE=2,mode='NELL'):
    if mode!='NELL':
        tree=typetree()
    ent_ser,rel_ser=load_paths(paths)
    with open(ent2type,'r') as fin:
        ent2type=json.load(fin)

    type_num={}
    for ent in ent_ser:
        for e in ent:
            typelist=ent2type[e]

            for t in typelist:
                if t in type_num:
                    type_num[t]+=1
                else:
                    type_num[t]=1

    typeinfo=[]
    for ent in ent_ser:
        typeinfo_unit=[]
        for e in ent:
            #print('test ent:',e)
            typelist=ent2type[e]
            #print('Typelist:',typelist)
            if mode!='NELL':
                res=tree.filter_typelist(typelist)
                candidate_set=set([])
                for pp in res:
                    candidate_set.add(pp[-1])
            else:
                candidate_set=set(typelist.copy())
            if len(candidate_set)>MAX_TYPE:
                sorted_ty=[(i,type_num[i]) for i in candidate_set]
                sorted_ty=sorted(sorted_ty,key=lambda x:x[1],reverse=True)
                candidate_set=[ge[0] for ge in sorted_ty[:MAX_TYPE]]
            assert len(candidate_set)>0
            typeinfo_unit.append(candidate_set)
        typeinfo.append(typeinfo_unit)
    metapath={}
    for u,v in zip(typeinfo,rel_ser):
        path_l=len(v)+len(u)
        metapath_u=[0]*path_l
        for k,r in enumerate(v):
            metapath_u[k*2+1]=r
        for ty_u in product(*u):
            for k,t in enumerate(ty_u):
                metapath_u[k*2]=t
            try:
                sk=tuple(metapath_u)
                if sk in metapath:
                    metapath[sk]+=1
                else:
                    metapath[sk]=1
            except TypeError:
                print(tuple(metapath_u))
    sort_metapath=sorted(metapath.items(),key=lambda x:x[1],reverse=True)
    with open(save_path,'w') as fin:
        wrt_str=['\t'.join(k)+'\t'+str(v) for k,v in sort_metapath]
        wrt_str='\n'.join(wrt_str)
        fin.write(wrt_str)

def main():
    rootpath = './data/yagosmall/'
    task_relation = "<isCitizenOf>"
    savefolder = rootpath + task_relation
    if not os.path.exists(savefolder):
        os.mkdir(savefolder)


    fact_path = rootpath + 'yagoFacts_DONE'
    ent2type_path = rootpath +'yagoFacts_ent2type.json'

    preprocessing(fact_path, ent2type_path, savefolder)
    train_rootpath=rootpath+task_relation+'/'
    train(task_relation,rootpath+task_relation+'/',epoch=5)
    write_test_pairs(task_relation, train_rootpath)
    paths = train_rootpath + task_relation + '.paths'
    ent2type = train_rootpath + 'ent2type.json'
    save_path = train_rootpath + task_relation + '.metapaths'
    generate_metapath(paths, ent2type, save_path, MAX_TYPE=2, mode='Yago')
def main_nell():
    rootpath = './data/NELL/'

    task_relation_set = ['concept:teamplaysagainstteam',
                         'concept:competeswith',
                         'concept:worksfor']
    for task_relation in task_relation_set:
        savefolder = rootpath + task_relation
        if not os.path.exists(savefolder):
            os.mkdir(savefolder)

        fact_path = rootpath + 'NELL_DONE'
        ent2type_path = rootpath + 'NELL_ent2type_DONE.json'
        train_rootpath=savefolder+'/'
        preprocessing(fact_path, ent2type_path, savefolder)
        train(task_relation,train_rootpath,epoch=5)

        write_test_pairs(task_relation, train_rootpath)
        paths=train_rootpath+task_relation+'.paths'
        ent2type=train_rootpath+'ent2type.json'
        save_path=train_rootpath+task_relation+'.metapaths'
        generate_metapath(paths,ent2type,save_path,MAX_TYPE=1,mode='NELL')
if __name__=='__main__':
    main_nell()

