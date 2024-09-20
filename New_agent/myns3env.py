import numpy as np
import gym
from gym import spaces
from ns3gym import ns3env
import csv
import time
from DDQN_class import DDQNAgent
reward1=[]
reward2=[]
reward3=[]
reward4=[] #New penalized throughput
reward5=[] #Minimum throughput
reward6=[] #counter 1
reward7=[] #counter 2
reward8=[] #counter 3
reward9=[]
reward10=[]
reward11=[]
data = []
step_throughput =0
MIMO =[]           #mimo action
COI=[]
ActivUes=[]
avg_reward1=[]
avg_reward2=[]
avg_reward3=[]
avg_reward4=[]
avg_reward5=[]
avg_reward6=[]
avg_reward7=[]
avg_reward8=[]
avg_reward9=[]
avg_reward10=[]
avg_reward11=[]
avg_MIMO=[]
avg_COI=[]
MCS2CQI=np.array([1,2,3,3,3,4,4,5,5,6,6,6,7,7,8,8,8,9,9,9,10,10,10,11,11,12,12,13,14])
step_MIMO=1 # MIMO ON/OFF (in the discrete set {0, 1})
c = [0,0,0,0,0,0]
actv_cell=[]



class myns3env(gym.Env):
  """
  Custom Environment that follows gym interface.
  This is a simple env where the agent must learn to go always left.
  """

  def __init__(self,):
    super(myns3env, self).__init__()
    port=2211
    simTime= 51
    stepTime=1
    startSim=0
    seed=3
    simArgs = {"--duration": simTime,}
    debug=True
    self.max_env_steps = 249
    self.env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim, simSeed=seed, simArgs=simArgs, debug=debug)
    self.env._max_episode_steps = self.max_env_steps
    self.time_var="t_{}".format(int(time.time()))

    self.Cell_num=6  #Cell number changed
    self.max_throu=30
    self.Users=40
    self.state_dim = self.Cell_num*5   #Add MIMO state
    self.state_short_dim = self.Cell_num*4
    self.state_short=[1]*self.state_short_dim
    #print("state_dim:{}".format(self.state_dim ))
    self.action_dim =  self.env.action_space.shape[0]
    self.CIO_action_bound =  6
    self.power_action_bound =  6#self.env.action_space.high
    
    self.action_DDQN=[]
    
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions, we have two: left and right

    self.action_space = spaces.Box(low=-1, high=1,
                                        shape=(self.action_dim,), dtype=np.float32)
    
    # The observation will be the coordinate of the agent
    # this can be described both by Discrete and Box space
    self.observation_space = spaces.Box(low=0, high=self.Users,
                                        shape=(self.state_dim,), dtype=np.float32)
    #print("action_space :", self.action_space)
    
    
 #   self.state_short = 24
    self.a_level = 2   #MIMO is ON or OFF
    self.a_num = 6   #2 cells
    self.DDQN_action_size = self.a_level**self.a_num
    self.agent = DDQNAgent(self.state_short_dim, self.DDQN_action_size)       
    #self.agent.load("DDQN_mu_0_lampda_2")            
    done = False
    
  def reset(self):
    """
    Important: the observation must be a numpy array :return: (np.array)
    """
    #start1 = time.process_time()
    print("reset                       reset                       reset                       reset                       reset                       ")
    state = self.env.reset()
    if state is None:#To avoid crashing the simulation if the handover failiure occured in NS-3 simulation
        state=[0] * self.state_dim
        return np.array(state)
    state1 = np.reshape(state['rbUtil'], [self.Cell_num, 1])#Reshape the matrix
    state2 = np.reshape(state['dlThroughput'],[self.Cell_num,1])
    state2_norm=state2/self.max_throu
    state3 = np.reshape(state['UserCount'], [self.Cell_num, 1])#Reshape the matrix
    state3_norm=state3/self.Users
    MCS_t=np.array(state['MCSPen'])
    state4=np.sum(MCS_t[:,:10], axis=1)
    state4=np.reshape(state4,[self.Cell_num,1])
    state5=np.zeros([self.Cell_num, 1], dtype = None, order = 'C')
    # To report other rewards
    R_rewards = np.reshape(state['rewards'], [11, 1])#Reshape the matrix
    R_rewards =[j for sub in R_rewards for j in sub]

    state  = np.concatenate((state1,state2_norm,state3_norm,state4,state5),axis=None)
    self.state_short = np.concatenate((state1,state2_norm,state3_norm,state4),axis=None) 
    #print("RB utilization:{}".format(np.transpose(state1)))
    #print("Cell throuhput:{}".format(np.transpose(state2)))
    #print("Users count:{}".format(np.transpose(state3)))
    #print("Edge user ratio:{}".format(np.transpose(state4)))
    #print("rewards:{}".format(np.transpose(R_rewards)))

    state = np.reshape(state, [self.state_dim,])###
    print("state after reset",state)
    self.state_short = np.reshape(self.state_short, [1,self.state_short_dim])###
    #print(" state.shape:{}".format( state.shape))
    #print("self.observation_space.shape:{}".format(self.observation_space.shape))
    #if (    state.shape ==self.observation_space.shape and np.all(state >= self.observation_space.low) and np.all(state <= self.observation_space.high)):
        #print("ok")
    #print("np.array(state):{}".format(np.array(state)))
    #print("reset time", time.process_time() - start1)
    return np.array(state)

  def step(self, action_TD3):
      
    global data
    #Call DDQN agent before TD3
    #global action_DDQN
    
    print("state_short", self.state_short)
    
    action_index = self.agent.act(self.state_short)
    self.action_DDQN=np.base_repr(action_index+int(self.a_level)**int(self.a_num),base=int(self.a_level))[-self.a_num:]# decoding the action index to the action vector
    self.action_DDQN=[int(self.action_DDQN[s]) for s in range(len(self.action_DDQN))]
    #action_DDQN=np.concatenate((np.zeros(a_num-len(action_DDQN)),action_DDQN),axis=None)
    self.action_DDQN=[step_MIMO*x for x in self.action_DDQN]#action vector
    #action_DDQN=[0,0,0,0,0,0]#SISO only
    #self.action_DDQN=[1,1,1,1,1,1]#MIMO only
    print("mimo_action" , self.action_DDQN)
    DDQN_action_total = sum(self.action_DDQN)  / len(self.action_DDQN)                                   # <<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>
    DDQN_action_sum = sum(self.action_DDQN)
    data = [DDQN_action_sum]
    
    
    
    
    
    print ("MIMO_ total ",sum(self.action_DDQN) )
    for ii in range(self.Cell_num):
        if self.action_DDQN[ii] == 1:
            c[ii] +=1
       
       
    print("counter is")
    print(c)
    ###############################################################
    
    #TD3 action
    action1=action_TD3[:11]*(self.CIO_action_bound)
    action2=action_TD3[11:17]*(self.power_action_bound)
   
    action_TD3  = np.concatenate((action1,action2),axis=None)
    
    ##############################################################
    
    #Whole action
    big_action = np.concatenate((action_TD3,self.action_DDQN),axis=None)
    
  
    next_state, reward, done, info = self.env.step(big_action)
    #print("next_state",next_state)
    print("Reward {}".format(reward))
    ###########################################################33
    
    #Check for correct process 
    ########################
    
    
    global reward1
    global reward2
    global reward3
    global reward4
    global reward5
    global reward6
    global reward7
    global reward8
    global reward9
    global reward10
    global reward11
    global MIMO
    global COI
    global ActivUes
    global step_throughput
    
    
    if next_state is None:#To avoid crashing the simulation if the handover failiure occured in NS-3 simulation 
    #################################################################################################################
        Result_row=[]
        with open('Rewards_testing' + self.time_var + '.csv', 'w', newline='') as rewardcsv:
            results_writer = csv.writer(rewardcsv, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
            Result_row.clear()
            Result_row=Result_row+reward1     
            results_writer.writerow(Result_row)
        rewardcsv.close()
        

####################################################################################################
        if len(reward1) % self.max_env_steps != 0:
            print("Interrupted episode,  after {} steps".format((len(reward1) % self.max_env_steps)))
            print("len(reward1):{}".format(len(reward1)))
            for e in range(len(reward1) % self.max_env_steps): 
              reward1.pop()
              
        if len(reward2) % self.max_env_steps != 0:
            print("len(reward2):{}".format(len(reward2)))
            for e in range(len(reward2) % self.max_env_steps): 
              reward2.pop()

        if len(reward3) % self.max_env_steps != 0:
            print("len(reward3):{}".format(len(reward3)))
            for e in range(len(reward3) % self.max_env_steps): 
              reward3.pop()
              
        if len(reward4) % self.max_env_steps != 0:
            print("len(reward4):{}".format(len(reward4)))
            for e in range(len(reward4) % self.max_env_steps): 
              reward4.pop()
              
        if len(reward5) % self.max_env_steps != 0:
            print("len(reward5):{}".format(len(reward5)))
            for e in range(len(reward5) % self.max_env_steps): 
              reward5.pop()
              
        if len(reward6) % self.max_env_steps != 0:
            print("len(reward6):{}".format(len(reward6)))
            for e in range(len(reward6) % self.max_env_steps): 
              reward6.pop()
         
        if len(reward7) % self.max_env_steps != 0:
            print("len(reward7):{}".format(len(reward7)))
            for e in range(len(reward7) % self.max_env_steps): 
              reward7.pop()
              
        if len(reward8) % self.max_env_steps != 0:
            print("len(reward8):{}".format(len(reward8)))
            for e in range(len(reward8) % self.max_env_steps): 
              reward8.pop()
              
        if len(reward9) % self.max_env_steps != 0:
            print("len(reward9):{}".format(len(reward9)))
            for e in range(len(reward9) % self.max_env_steps): 
              reward9.pop()
                    
        if len(reward10) % self.max_env_steps != 0:
            print("len(reward10):{}".format(len(reward10)))
            for e in range(len(reward10) % self.max_env_steps): 
              reward10.pop()
              
        if len(reward11) % self.max_env_steps != 0:
            print("len(reward11):{}".format(len(reward11)))
            for e in range(len(reward11) % self.max_env_steps): 
              reward11.pop()
              
        if len(COI) % self.max_env_steps != 0:
            print("len(COI):{}".format(len(COI)))
            for e in range(len(COI) % self.max_env_steps): 
              COI.pop()
              
        if len(ActivUes) % self.max_env_steps != 0:
            print("len(ActivUes):{}".format(len(ActivUes)))
            for e in range(len(ActivUes) % self.max_env_steps): 
              ActivUes.pop()
              
        if len(MIMO) % self.max_env_steps != 0:
            print("len(MIMO):{}".format(len(MIMO)))
            for e in range(len(MIMO) % self.max_env_steps): 
              MIMO.pop()  
                  
        if len(actv_cell) % self.max_env_steps != 0:
            print("len(actv_cell):{}".format(len(actv_cell)))
            for e in range(len(actv_cell) % self.max_env_steps): 
              actv_cell.pop()      
                              
        reward=0         # Handover failiure occured
        step_throughput =0
        done=True
        next_state=[0] * self.state_dim
        return np.array(next_state), reward, done, info
    
    #####################################################################
    state1 = np.reshape(next_state['rbUtil'], [self.Cell_num, 1])
    state2 = np.reshape(next_state['dlThroughput'],[self.Cell_num, 1])					
    state2_norm=state2/self.max_throu
    state3 = np.reshape(next_state['UserCount'], [self.Cell_num, 1])
    state3_norm=state3/self.Users
    MCS_t=np.array(next_state['MCSPen'])
    state4=np.sum(MCS_t[:,:10], axis=1)
    state4=np.reshape(state4,[self.Cell_num, 1])
    # To report other reward functions
    R_rewards = np.reshape(next_state['rewards'], [11, 1])               
    R_rewards =[j for sub in R_rewards for j in sub]
   

	    # Map MCS to CQI
    if (sum(state3)) == 0:
    	for i in range(self.Cell_num):
    		xx=state3 [i]
    		MCS_t[i,:] *= xx/self.Users
    else:
    	for i in range(self.Cell_num):
    		xx=state3 [i]
    		MCS_t[i,:] *= xx/sum(state3)#self.Users

        
    AVGMCS= [sum(x) for x in zip(*MCS_t)]
    AVGMCS=np.reshape(AVGMCS,[1, 29])
    AVG_CQI=np.sum(AVGMCS*MCS2CQI)

###############################################################################

#DDQN learning
##############
    
    next_state  = np.concatenate((state1,state2_norm,state3_norm,state4),axis=None)
    next_state = np.reshape(next_state, [1, self.state_short_dim])
    self.agent.remember(next_state, action_index, reward, next_state, done) # Add to the experience buffer
    
    
    if len(self.agent.memory)/25 > self.agent.batch_size:
        loss = self.agent.replay(self.agent.batch_size)
    
    
    ##################################################################################33
        
    state5=np.reshape(self.action_DDQN,[self.Cell_num, 1])
  
    step_throughput = R_rewards[0]
    
    reward1.append(R_rewards[0])
    reward2.append(R_rewards[1])
    reward3.append(R_rewards[2])
    reward4.append(R_rewards[3])
    reward5.append(R_rewards[4])
    reward6.append(R_rewards[5])
    reward7.append(R_rewards[6])
    reward8.append(R_rewards[7])
    reward9.append(R_rewards[8])
    reward10.append(R_rewards[9])
    reward11.append(R_rewards[10])
    MIMO.append(DDQN_action_total)
    COI.append(AVG_CQI)
    actv_cell.append(data)
    
    if (sum(state3)) == 0:
        ActivUes.append(np.array(self.Users).reshape(1))
    else:
        ActivUes.append(sum(state3))
           
    print("ActivUes:{}".format(ActivUes[- 1]))

    if len(reward1) % self.max_env_steps == 0:
        self.agent.save("DDQN_mu_0_lampda_2")
        avg_reward1 = np.array(reward1).reshape(-1, self.max_env_steps).mean(axis=1)
        avg_reward2 = np.array(reward2).reshape(-1, self.max_env_steps).mean(axis=1)
        avg_reward3 = np.array(reward3).reshape(-1, self.max_env_steps).mean(axis=1)
        avg_reward4 = np.array(reward4).reshape(-1, self.max_env_steps).mean(axis=1)
        avg_reward5 = np.array(reward5).reshape(-1, self.max_env_steps).mean(axis=1)
        avg_reward6 = np.array(reward6).reshape(-1, self.max_env_steps).mean(axis=1)
        avg_reward7 = np.array(reward7).reshape(-1, self.max_env_steps).mean(axis=1)
        avg_reward8 = np.array(reward8).reshape(-1, self.max_env_steps).mean(axis=1)
        avg_reward9 = np.array(reward9).reshape(-1, self.max_env_steps).mean(axis=1)
        avg_reward10 = np.array(reward10).reshape(-1, self.max_env_steps).mean(axis=1)
        avg_reward11 = np.array(reward11).reshape(-1, self.max_env_steps).mean(axis=1)
        avg_MIMO = np.array(MIMO).reshape(-1, self.max_env_steps).mean(axis=1)
        avg_COI1 = np.array(COI).reshape(-1, self.max_env_steps).mean(axis=1)
        avg_ActivUes = np.array(ActivUes).reshape(-1, self.max_env_steps).mean(axis=1)
        std_ActivUes = np.array(ActivUes).reshape(-1, self.max_env_steps).std(axis=1)
        

    if len(reward1) % self.max_env_steps == 0:
        Result_row=[]
        with open('Rewards_testing' + self.time_var + '.csv', 'w', newline='') as rewardcsv:
            results_writer = csv.writer(rewardcsv, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
            Result_row.clear()
            Result_row=Result_row+reward1     #row0   sum of throughput
            results_writer.writerow(Result_row)
            Result_row.clear()
            Result_row=Result_row+reward2     #row1   penalized throughput
            results_writer.writerow(Result_row)
            Result_row.clear()
            Result_row=Result_row+reward3     #row2    non blocked users
            results_writer.writerow(Result_row)
            Result_row.clear()
            Result_row=Result_row+reward4     #row3   new penalized throughput
            results_writer.writerow(Result_row)
            Result_row.clear()
            Result_row=Result_row+COI      #row4     CQI
            results_writer.writerow(Result_row)
            Result_row.clear()
            Result_row=Result_row+ActivUes    #row5    Active users
            results_writer.writerow(Result_row)
            Result_row.clear()
            Result_row=Result_row+avg_reward1.tolist()    #row6    avg sum of throughput
            results_writer.writerow(Result_row)
            Result_row.clear()
            Result_row=Result_row+avg_reward2.tolist()     #row7    avg penalized throughput
            results_writer.writerow(Result_row)
            Result_row.clear()
            Result_row=Result_row+avg_reward3.tolist()     #row8   avg non blocked users
            results_writer.writerow(Result_row)
            Result_row.clear()
            Result_row=Result_row+avg_reward4.tolist()       #row9 ( avg new penalized throughput)
            results_writer.writerow(Result_row)
            Result_row.clear()
            Result_row=Result_row+avg_COI1.tolist()     #row10     avg CQI
            results_writer.writerow(Result_row)
            Result_row.clear()
            Result_row=Result_row+avg_ActivUes.tolist()     #row11   avg active users
            results_writer.writerow(Result_row)
            Result_row.clear()
            Result_row=Result_row+std_ActivUes.tolist()     #row12
            results_writer.writerow(Result_row)
            Result_row.clear()
            Result_row=Result_row+reward5     #row13    minimum throughput
            results_writer.writerow(Result_row)
            Result_row.clear()
            Result_row=Result_row+avg_reward5.tolist()       #row14   averge minimum throughput
            results_writer.writerow(Result_row)
            Result_row.clear()
            Result_row=Result_row+reward6     #row15    counter1
            results_writer.writerow(Result_row)
            Result_row.clear()
            Result_row=Result_row+avg_reward6.tolist()       #row16   averge counter1
            results_writer.writerow(Result_row)
            Result_row.clear()
            Result_row=Result_row+reward7     #row17    counter 2
            results_writer.writerow(Result_row)
            Result_row.clear()
            Result_row=Result_row+avg_reward7.tolist()       #row18   averge counter 2
            results_writer.writerow(Result_row)
            Result_row.clear()
            Result_row=Result_row+reward8     #row19    counter 3
            results_writer.writerow(Result_row)
            Result_row.clear()
            Result_row=Result_row+avg_reward8.tolist()       #row20   averge counter 3
            results_writer.writerow(Result_row)
            Result_row=Result_row+reward9     #row20    counter4
            results_writer.writerow(Result_row)
            Result_row.clear()
            Result_row=Result_row+avg_reward9.tolist()       #row20   averge counter4
            results_writer.writerow(Result_row)
            Result_row.clear()
            Result_row=Result_row+reward10     #row21    counter 5
            results_writer.writerow(Result_row)
            Result_row.clear()
            Result_row=Result_row+avg_reward10.tolist()       #row21   averge counter 5
            results_writer.writerow(Result_row)
            Result_row.clear()
            Result_row=Result_row+reward11    #row22    counter 6
            results_writer.writerow(Result_row)
            Result_row.clear()
            Result_row=Result_row+avg_reward11.tolist()       #row22   averge counter 6
            results_writer.writerow(Result_row)
            Result_row.clear()
            Result_row=Result_row+actv_cell    #row23  averge MIMO ACtion total
            results_writer.writerow(Result_row)
        rewardcsv.close()
    print("action:{}".format((big_action)))
    print("R_rewards:{}".format((R_rewards)))
    #print("done:{}".format((done)))

    #print("reward1 :{}".format(reward1))
    #print("reward2 :{}".format(reward2))
    #print("reward3 :{}".format(reward3))

    print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-")
    #R_reward1,R_reward2,R_reward3 = np.reshape(state['rewards'], [3, 1])#Reshape the matrix
    big_state  = np.concatenate((state1,state2_norm,state3_norm,state4,state5),axis=None)
    big_state = np.reshape(big_state, [self.state_dim,])
    print("big state",big_state)

    self.state_short = np.concatenate((state1,state2_norm,state3_norm,state4),axis=None)
    self.state_short = np.reshape(self.state_short, [1,self.state_short_dim])

    
    return np.array(big_state), step_throughput, done, info

  def render(self, mode='console'):
    print("......")
   # return step_throughput



  def close(self):
    pass

