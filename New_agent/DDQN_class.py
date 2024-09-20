#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 00:22:09 2021

@author: mariam
"""

import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
#from keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import schedules
from keras import backend as K

import tensorflow as tf
import pdb
#import tensorflow.contrib.slim as slim
import matplotlib as mpl                  #for plotting
import matplotlib.pyplot as plt
from tensorflow import keras
from ns3gym import ns3env                 #for interfacing NS-3 with RL
from keras.layers import Dense, Dropout, Activation
import csv
import time

stepTime=0.2
startSim=0
Usersnum=41
seed=3
simArgs = {}
debug=True
#the step of discrete action vector
step_MIMO=2 # MIMO mode step in the discrete set {1, 3}
Result_row=[]
Rew_ActIndx=[]
MCS2CQI=np.array([1,2,3,3,3,4,4,5,5,6,6,6,7,7,8,8,8,9,9,9,10,10,10,11,11,12,12,13,14])# To map MCS indexes to CQI

max_env_steps = 299  #Maximum number of steps in every episode
class DDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)
        self.gamma = 0.95    # Discount rate
        self.epsilon = 1 # At the begining
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.001     #0.001 changed
      #  self.lr_schedule = schedules.ExponentialDecay(initial_learning_rate=1e-2,decay_steps=10000,decay_rate=0.9)                  #ADDED
        self.Prev_Mean=0 #initial mean of the targets (for target normalization)
        self.Prev_std=1 #initial std of the targets (for target normalization)
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.batch_size = 32
        #self.Find_Best_Action(self, next_state,a_level,a_num)


    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
   
        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))								#"lr_schedule" instead of "learning_rate"
        return model

    def update_target_model(self):
        # copy weights from the CIO selection network to target network
        self.target_model.set_weights(self.model.get_weights())
        
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def act(self, state):
        act_values = self.model.predict(state)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        return np.argmax(act_values[0])  # returns action


    def replay(self, batch_size):
           
        minibatch = random.sample(self.memory, batch_size)

        target_A=[]#for batch level target normalization

        for state, action, reward, next_state, done in minibatch:#calculate the target array
            a = self.model.predict(next_state)[0]
            t = self.target_model.predict(next_state)[0]
            b =t[np.argmax(a)]#needs de_normalization
            b *=self.Prev_std
            b =+self.Prev_Mean
            target_A.append(reward + self.gamma * b)
        
        mean_MB= np.mean(np.asarray(target_A), axis=0) #mean of the targets in this mini-batch
        std_MB= np.std(np.asarray(target_A), axis=0) # std of the targets in this mini-batch

        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                tg = reward
            else:
                a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0] # DDQN feature

                b =t[np.argmax(a)]#needs de_normalization
                b *=self.Prev_std # denormalize the future reward by the mean and std of the previous mini-batch
                b =+self.Prev_Mean # denormalized future reward
                tg = reward + self.gamma * b  # 
                tg -= mean_MB
                tg /= std_MB #normalized target
                self.Prev_std = std_MB
                self.Prev_Mean= mean_MB
            target[0][action]=tg
            history = self.model.fit(state, target, epochs=1, verbose=0)# training
        loss = history.history['loss'][0]
        #print("loss",loss)
        if self.epsilon > self.epsilon_min: #To balance the exploration and exploitation
            self.epsilon *= self.epsilon_decay
        return loss


    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
