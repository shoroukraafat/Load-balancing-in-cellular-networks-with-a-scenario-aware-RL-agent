import os
import time
import gym
import numpy as np
import matplotlib.pyplot as plt

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

import tensorflow.compat.v1 as tf
physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

#gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
#tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=268)])   
    
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        print("Steps: {}".format(self.num_timesteps))

        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True

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

model = TD3(MlpPolicy, env, action_noise=action_noise, verbose=1, tensorboard_log="./TD3_ped_veh_tensorboard/",learning_starts=100)

# Create the callback: check every 1000 steps
callback = SaveOnBestTrainingRewardCallback(check_freq=250, log_dir=log_dir) 
# Train the agent
time_steps = 15000
episode_steps =250
episode_number = int(time_steps/episode_steps)
model.learn(total_timesteps=int(time_steps), callback=callback)
model.save("TD3_mu0_lambda_0_40_User".format(int(time.time())))

# Log results 
results_plotter.plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "TD3")
plt.show()

print("######################End of training#####################")
'''

# Test the agent
print("######################Testing#####################")
episode_rewards = []
episode_throughputs = []
Step_rewards = []
Step_throughputs = []
for i in range(1):
    reward_sum = 0
    throughput_sum = 0
    obs = env.reset()
    for j in range(episode_steps):
        print("Test: Step : {} | Episode: {}".format(j, i))
        action, _states = model.predict(obs)
        print(action)
        obs, rewards, dones, info = env.step(action)
        #throughput = env.ret_throughput();
        Step_rewards.append(rewards)
        #Step_throughputs.append(throughput)
        reward_sum += rewards
        #throughput_sum += throughput
    episode_rewards.append(reward_sum)
    #episode_throughputs.append(throughput_sum)
episode_rewards = [x / episode_steps for x in episode_rewards]
#episode_throughputs = [y / episode_steps for y in episode_throughputs]
#DDQN_action_sum = DDQN_action_sum/(6*episode_steps )

#for k in range(episode_steps):
#       episode_rewards.append(reward_sum)
#episode_rewards = [x / episode_steps for x in episode_rewards]

episode_rewards_0 = []
Step_rewards0 = []
episode_throughputs_0 = []
Step_throughputs0 = []

for i in range(1):
    reward_sum0 = 0
    throughput_sum0 = 0
    obs = env.reset()
    for j in range(episode_steps):
        action_0=[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
       
        print("Baseline: Step : {} | Episode: {}".format(j, i))
        obs, rewards, dones, info = env.step(action_0)
        #throughput = env.ret_throughput();
        reward_sum0 += rewards
        #throughput_sum0 += throughput
        Step_rewards0.append(rewards)
        #Step_throughputs0.append(throughput)


    episode_rewards_0.append(reward_sum0)
    #episode_throughputs_0.append(throughput_sum0)

episode_rewards_0 = [x / episode_steps for x in episode_rewards_0]
#episode_throughputs_0 = [y / episode_steps for y in episode_throughputs_0]
#for k in range(episode_steps):
#       episode_rewards_0.append(reward_sum0)

#episode_rewards_0 = [x / episode_steps for x in episode_rewards_0]
#print("MIMO action percentage {}".format(DDQN_action_sum))
Result_row=[]
with open('R1_train' + 'TD3_step' + format(int(time.time()))+'.csv', 'w', newline='') as BSCSV:

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
              
BSCSV.close()





fig1, ax = plt.subplots()
ln1, = plt.plot(episode_rewards, label='TD3 Average')
ln1, = plt.plot(episode_rewards_0, label='Baseline Average')
ln1, = plt.plot(Step_rewards, label='TD3')
ln1, = plt.plot(Step_rewards0, label='Baseline')

legend = ax.legend(loc='upper left', shadow=True, fontsize='x-small')
plt.xlabel("Step")
plt.ylabel("Average Overall throughput")
plt.title('Comparing step reward: TD3 vs. baseline')
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


'''









