import os
import time
import gym
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3.common import results_plotter
from VecMonitor import VecMonitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
#from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from myns3env import myns3env
import csv

time_steps = 5000
episode_steps = 250
episode_number = int(time_steps/episode_steps)


# Create log dir
log_dir = "tmp_{}/".format(int(time.time()))
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
# Instantiate the env
env = myns3env()
# wrap it
env = make_vec_env(lambda: env, n_envs=1)

env = VecMonitor(env, log_dir)
#env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=100.)
# the noise objects for TD3

n_actions = env.action_space.shape[-1]

action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = TD3(MlpPolicy, env, action_noise=action_noise, verbose=1, tensorboard_log="./TD3_ped_veh_tensorboard/")

#model = TD3.load ("TD3_ped_veh_r1_1680426721")         #mew =5 
#model = TD3.load ("TD3_ped_veh_r1_1679984265")      # mew = 2
agent_dec = env.envs[0].action_DDQN

# Test the agent
print("######################Testing#####################")



DDQN_vec = []
episode_rewards_0 = []
Step_rewards0 = []
episode_throughputs_0 = []
Step_throughputs0 = []



for i in range(10):
    reward_sum0 = 0
    throughput_sum0 = 0
    obs = env.reset()
    for j in range(episode_steps):
    
    	#agent1_dec = env.envs[0].action_DDQN
        #action_0=[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
        #model= TD3.load('/mnt/Storage/repos/ns-3-allinone/ns-3.30/scratch/New_agent/mimo_only_mu0_lambda_0_40_User.zip', env=env, action_noise=action_noise, verbose=1) 
        model = TD3.load('/mnt/Storage/repos/ns-3-allinone/ns-3.30/scratch/New_agent/TD3_orig_mu0_lambda_0_40_User.zip', env=env, action_noise=action_noise, verbose=1)
        agent1_dec = env.envs[0].action_DDQN
        action_0, _states = model.predict(obs)
        print("Single: Step : {} | Episode: {}".format(j, i))
        obs, rewards, dones, info = env.step(action_0)
        #throughput = env.ret_throughput();
        reward_sum0 += rewards
        #throughput_sum0 += throughput
        Step_rewards0.append(rewards)
        DDQN_vec.append(sum(agent1_dec))
        episode_rewards_0.append(reward_sum0)
episode_rewards_0 = [x / episode_steps for x in episode_rewards_0]


episode_rewards = []
episode_throughputs = []
Step_rewards = []
Step_throughputs = []
#DDQN_vec = []

for i in range(10):
     reward_sum = 0
     obs = env.reset()
     for j in range(episode_steps):  
     
         agent1_dec = env.envs[0].action_DDQN
         #print("mimo_" , agent1_dec)
         
         if agent1_dec == [1, 1, 1, 1, 1, 1]:
      
          model = TD3.load('/mnt/Storage/repos/ns-3-allinone/ns-3.30/scratch/New_agent/mimo_only_mu0_lambda_0_40_User.zip', env=env, action_noise=action_noise, verbose=1)
          print("MIMO model is chosen")
         else:
          model= TD3.load('/mnt/Storage/repos/ns-3-allinone/ns-3.30/scratch/New_agent/TD3_orig_mu0_lambda_0_40_User.zip', env=env, action_noise=action_noise, verbose=1) 
          print("All model is chosen")
         print("Hybrid: Step : {} | Episode: {}".format(j, i))
         if(j % 10 == 0):
          action, _states = model.predict(obs)
          print(action)
         obs, rewards, dones, info = env.step(action)
         Step_rewards.append(rewards)
         DDQN_vec.append(sum(agent1_dec))
         reward_sum += rewards
         episode_rewards.append(reward_sum)
episode_rewards = [x / episode_steps for x in episode_rewards]


#print('singe model average Reward:{} '.format(episode_rewards_0))



Result_row=[]
with open('R1_test_lampda0' + 'TD3_step' + format(int(time.time()))+'.csv', 'w', newline='') as BSCSV:

                     results_writer = csv.writer(BSCSV, delimiter=';', quotechar=';', quoting=csv.QUOTE_MINIMAL)

                     Result_row.clear()

                     Result_row=Result_row+episode_rewards_0

                     results_writer.writerow(Result_row)

                     Result_row.clear()

                     Result_row=Result_row+Step_rewards0

                     results_writer.writerow(Result_row)

                     Result_row.clear()

                     Result_row=Result_row+episode_rewards

                     results_writer.writerow(Result_row)

                     Result_row.clear()

                     Result_row=Result_row+Step_rewards

                     results_writer.writerow(Result_row)
              
                     Result_row.clear()

                     Result_row=Result_row+DDQN_vec

                     results_writer.writerow(Result_row)
BSCSV.close()





fig1, ax = plt.subplots()
#ln1, = plt.plot(np.repeat(step_rewards,episode_steps,axis=0), label='All traied model')
#ln1, = plt.plot(np.repeat(step_rewards0,episode_steps,axis=0), label='MIMO trained model')
ln1, = plt.plot(Step_rewards, label='Alternating between 2 TD3 agents')
ln1, = plt.plot(Step_rewards0, label='Conventional Single TD3 agent')
plt.ylim(20, 35)
legend = ax.legend(loc='upper right', shadow=True, fontsize='x-small')
plt.xlabel("Step")
plt.ylabel("Average Overall throughput")
plt.title('Comparing step reward: All cases model vs. MIMO only model')
plt.savefig('TD3_0CIO_{}.png'.format(int(time.time())))




#fig2, ax2 = plt.subplots()
#ln2, = plt.plot(episode_throughputs, label='TD3 Average')
#ln2, = plt.plot(episode_throughputs_0, label='Baseline Average')
#ln2, = plt.plot(Step_throughputs, label='TD3')
#ln2, = plt.plot(Step_throughputs0, label='Baseline')

#legend = ax2.legend(loc='upper left', shadow=True, fontsize='x-small')
#plt.xlabel("Step")
#plt.ylabel("Average Overall Throughput")
#plt.title('Comparing step throughput: TD3 vs. baseline')
#plt.savefig('TD3_0CIO2_{}.png'.format(int(time.time())))




