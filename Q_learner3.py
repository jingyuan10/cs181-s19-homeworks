import numpy as np
import math
import random
from SwingyMonkey import SwingyMonkey
import csv

import pygame as pg



# Redefine distances and grid
# 1. dhead = the vertical distance from the head of the monkey to the top of the tree gap -- (state['tree']['top']-state['monkey']['top'])/dhead_binsize
# 2. tbot = the vertical distance the bottom of the tree gap to ground -- state['tree']['bot']/tbot_binsize    #range(0-200)
# 3. dh = the horizontal distance from the monkey to the next tree -- state['tree']['dist']/dv_binsize
# 4. v = the volecity of the monkey -- state['monkey']['vel'] /v_binsize

# Added customized alpha 

class QLearner3:

    def __init__(self):
        self.dhead_range=(-50, 400)
        self.dhead_binsize=30
        self.dhead_binnum=int((self.dhead_range[1]-self.dhead_range[0])/self.dhead_binsize)
        self.g_range=(1,5)
        self.g_binsize=0.5
        self.g_binnum=int((self.g_range[1]-self.g_range[0])/self.g_binsize)
        self.dh_range=(-150, 450)
        self.dh_binsize=20
        self.dh_binnum=int((self.dh_range[1]-self.dh_range[0])/self.dh_binsize)
        self.v_range=(-50,50)
        self.v_binsize=5
        self.v_binnum=int((self.v_range[1]-self.v_range[0])/self.v_binsize)

        # hyperparameters
        self.alpha = 0.6
        self.gamma = 0.6
        self.epsilon = 0.001

        # state parameters
        self.current_action = None
        self.current_state  = None
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

        self.epoc=0 
        self.iterr = 0


        # dimension of Q
        self.dim = (self.dhead_binnum+1,self.g_binnum+1,self.dh_binnum+1,self.v_binnum+1,2)
        #self.dim = (self.dhead_binnum+1,self.dh_binnum+1,self.v_binnum+1,2)
        self.Q = np.zeros(self.dim)
        self.k = np.ones(self.dim)

    def reset(self):
        self.current_action = None
        self.current_state  = None
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        #self.iter += 1
        self.epoc += 1
        
    def getstate(self,state):
        #should return the bin number of the state (dhead,tbot,dh,v)
        dhead=int(math.floor((state['tree']['top']-state['monkey']['top'])/self.dhead_binsize))
        g=int(math.floor(state['gravity']/self.g_binsize))
        dh=int(math.floor(state["tree"]["dist"]/self.dh_binsize))
        v=int(math.floor(state["monkey"]["vel"]/self.v_binsize))
        return (dhead,g,dh,v)
        #return (dhead,dh,v)
    

    def action_callback(self,state):
        '''Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.'''

        # epsilon-greedy policy

        #random action = generate a random action by randomly sample a number from 0 to 1 
        #With probability epsion, select the random action, and 
        #with probability 1-epsion select a greedy action (choose the max in the Q table)
        
        #self.current_action = random.choice((0,1))
        #self.current_state  = state
        if self.last_state == None:
            next_action = random.choice((0,1))
        else:
            if (random.random()<self.epsilon):
                next_action = random.choice((0,1))
            else:
                next_action = np.argmax(self.Q[self.getstate(state)])

        s  = self.getstate(state)
        a  = (self.last_action,)
        self.k[s + a] += 1
        self.alpha = 1/self.k[s + a]
        self.last_state  = self.current_state
        self.last_action = next_action
        self.current_state = state
        
        return next_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        if (self.last_state != None) and (self.current_state != None) and (self.last_action != None):
            st = self.getstate(self.last_state)
            st_1  = self.getstate(self.current_state)
            at  = (self.last_action,)

            #if self.iterr < 100:
                #alpha = self.alpha
            #else:
                #alpha = self.alpha*0.1

            #update Q
            alpha=self.alpha
            #print alpha
            #print alpha * (reward + self.gamma * np.max(self.Q[st_1]) - self.Q[st + at] )
            #print st+at
            self.Q[st + at] = self.Q[st + at] + alpha * (reward + self.gamma * np.max(self.Q[st_1]) - self.Q[st + at] )

        #self.last_reward = reward

        
        
def testgame(iters=100,show=True):

    learner = QLearner3()

    highestscore = 0
    avgscore = 0
    learner.alpha=0.2
    learner.gamma=0.6
    alpha = learner.alpha
    gamma = learner.gamma
    with open("test_gravity.csv","w",newline='') as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerow(["alpha","gamma","epoch","highest","average","score","q"])

    for ii in range(iters):

        learner.epsilon = 1/(ii+1)

        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,            # Don't play sounds.
                             text="Epoch %d" % (ii), # Display the epoch on screen.
                             tick_length=1,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        score = swing.get_state()['score']
        highestscore = max([highestscore, score])
        avgscore = (ii*avgscore+score)/(ii+1)
        q=round(float(np.count_nonzero(learner.Q))*100/learner.Q.size,3)
        
        if show==True:
            print ("epoch:",ii, "highest:", highestscore, "current score:", score, "average:", avgscore, "% of Q mx filled:", q )
        with open("test_gravity.csv","a+",newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows([[alpha, gamma,ii,highestscore,avgscore,score,q]])   
       
        # Reset the state of the learner.
        learner.reset()
    
    pg.quit()
    return avgscore,highestscore,score

testgame(iters=5000,show=False)