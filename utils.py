import torch
import random
import numpy as np
import logging
MARK_num=15
def setlogger():
    logger = logging.getLogger('RL4HIN')
    PrintStream = logging.StreamHandler()
    Debugwrite = logging.FileHandler(filename='./log/debug.log',mode='w')
    logger.setLevel(logging.DEBUG)
    PrintStream.setLevel(level=logging.INFO)
    Debugwrite.setLevel(level=logging.DEBUG)
    fil=logging.Filter()
    fil.filter=lambda x:True if x.levelno>=MARK_num else False
    format = logging.Formatter('%(asctime)s -%(funcName)s- %(levelname)s - %(message)s')
    Debugwrite.setFormatter(format)
    PrintStream.setFormatter(format)
    Debugwrite.addFilter(fil)
    logger.addHandler(PrintStream)
    logger.addHandler(Debugwrite)

    def Mark(record):
        return logger.log(MARK_num,record)

    logger.MARK=Mark
    return logger
logger=setlogger()
devicelist=['cpu','cuda:0']
global_device=devicelist[1]
def one_hot(input,num_class):
    if type(input)==int:
        res=[1 if i==input else 0 for i in range(num_class)]
        return res
    elif type(input)==list:
        res=[one_hot(i,num_class) for i in input]
        return res
    else:
        print(input,num_class)
        raise KeyError


def normalize(x,norm=2):
    if norm==2:
        no=sum([i*i for i in x])**0.5
        return [i/no for i in x]
    elif norm==1:
        pass