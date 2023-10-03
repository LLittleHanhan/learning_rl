'''
s 0 0 x 0
0 x 0 0 0
0 0 x 0 x
0 x 0 0 x
x 0 0 0 t
'''
import os
import numpy as np
import time
import random

action =np.array([[-1,0],[1,0],[0,-1],[0,1]])

map = np.array([[0,0,0,1,0],[0,1,0,0,0],[0,0,1,0,1],[0,1,0,0,1],[1,0,0,0,0]])
value = np.array([[0,0,0,-100,0],[0,-100,0,0,0],[0,0,-100,0,-100],[0,-100,0,0,-100],[-100,0,0,0,100]])
q_table = np.zeros([5,5,4],dtype=float)

learning_rate = 1
gamma = 0.9
epsilon = 0.9


def right_lo(lst):
    if -1<lst[0]<5 and -1<lst[1]<5: return True
    else: return False

def create_map():
    pass

def choose_max_action(x,y):
    val_lst = list(q_table[x][y])
    max_lst = []
    maxv = max(val_lst)
    for i,val in enumerate(val_lst):
        if val == maxv and right_lo(action[i]+np.array([x,y])):
            max_lst.append(i)
    act = random.choice(max_lst)
    return act

def choose_random_action(x,y):
    flag = True
    while flag:
        act = random.choice([0,1,2,3])
        if right_lo(action[act]+np.array([x,y])):
            flag = False
    return act

def update_qtable(x,y):
    radm = random.random()
    if radm < epsilon:
        act = choose_max_action(x,y) 
    else:
        act = choose_random_action(x,y)
    
    nlo = np.array([x,y])+action[act]
    nx = nlo[0]
    ny = nlo[1]
    nact = choose_max_action(nx,ny)
    # print("value",value[x][y])
    q_table[x][y][act] = q_table[x][y][act]+learning_rate*(value[nx][ny]+gamma*q_table[nx][ny][nact]-q_table[x][y][act])
    return nx,ny


def prnt(x,y):
    temp = map[x][y]
    map[x][y]=5
    print(map)
    time.sleep(0.5)
    os.system('cls')
    map[x][y]=temp


def q_learning():
    x=y=0
    while True:
        x,y = update_qtable(x,y)
        prnt(x,y)
        if value[x][y]==100:
            print("success")
            x=y=0
            continue
        if value[x][y]==-100:
            print("fail")
            x=y=0
            continue
        
if __name__=="__main__":
    q_learning()