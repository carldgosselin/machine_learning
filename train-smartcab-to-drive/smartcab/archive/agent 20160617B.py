import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

# for graphics and table views
import pylab as pl
import pandas as pd

# to create groupings by trial e.g. capture # of penalties per trial
import itertools
import operator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)	# sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'	# override color
        self.planner = RoutePlanner(self.env, self)	# simple route planner to get next_waypoint
        
        # TODO: Initialize any additional variables here
        self.actions = [None, 'forward', 'left', 'right']	# initialize all possible actions
        self.q_table = {}	# initialize q_table dictionary
        
        # Statistics - to measure performance
        self.num_moves = 0
        self.num_moves_total = 0
        self.penalty = 0
        self.group_penalties_per_trial = []
        self.group_number_of_moves_per_trial = []
        self.add_up_penalty_score = 0
        self.show_penalties = []
        self.penalties_to_rewards_progression = []
        self.reach_dest = 0
        self.trial_count = 0
        self.rewardTotalForTrip = 0
        self.reward_total_100trials = 0
        self.penalties_to_rewards_ratio = 0
        self.run_graph_once = 0
        
        # Epsilon, alpha, gamma
        self.epsilon_explore_vs_exploit = 0.15		# amount of exploration to do prior to exploitation (exploiting values in Q-table)
        self.alpha_learning_rate = 0.7				# relates to the immediate reward. 0 = no learning, 1  = full learning - overrides past learnings
        self.gamma_future_reward_discount = 0.2		# determines the importance of future rewards. 0 = focuses on immediate reward, 1 = strives for long-term high rewards

    def reset(self, destination=None):
        self.planner.route_to(destination)
        
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.num_moves = 0
        self.penalty = 0
        self.rewardTotalForTrip = 0
        self.trial_count += 1
        self.epsilon_explore_vs_exploit = 0.2
        
        # Decay epsilon rate as trial count increases
        self.epsilon_explore_vs_exploit = self.epsilonRateDecay(self.epsilon_explore_vs_exploit, self.trial_count)
        print "epsilon rate (decaying):", self.epsilon_explore_vs_exploit
        
        # Decay alpha rate as trial count increases
        self.alpha_learning_rate = self.alphaRateDecay(self.alpha_learning_rate, self.trial_count)
        print "alpha rate (decaying):", self.alpha_learning_rate

    def epsilonRateDecay(self, current_rate, trial_count):
        # Decay the epsilon rate as trial runs approach 100
        n_trials_plus_1 = 100 + 1  # increased by one to prevent division by zero on last run
        num2 = 1.0/float(n_trials_plus_1 - trial_count)
        decayed_rate = float(current_rate) - float(num2)
        if decayed_rate <= 0:
        	return 0
        else:
        	return decayed_rate

    def alphaRateDecay(self, current_rate, trial_count):
        # Decay the alpha rate as trial runs approach 100
        n_trials_and_then_some = 100 + 50  # increased number as rate decayed too quickly
        num2 = 1.0/float(n_trials_and_then_some - trial_count)
        decayed_rate = float(current_rate) - float(num2)
        if decayed_rate <= 0:
        	return 0
        else:
        	return decayed_rate
   
    def update(self, t):

        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = self.getState(inputs)	# New function. returns tuple ordered list
        
        # TODO: Select action according to your policy
        # action = None		
        
        # Epsilon-greedy - explore vs exploit
        if random.random() < self.epsilon_explore_vs_exploit:
        	action = random.choice(self.actions) # explore
        else:
        	action = self.getAction(self.state) # exploit
               
        # Execute action and get reward
        reward = self.env.act(self, action)
        self.rewardTotalForTrip += reward
        self.reward_total_100trials += reward
             
        # TODO: Learn policy based on state, action, reward
        thisQ = self.getQ(self.state, action)  # returns 0 if state/action tuple not yet in q_table
        nextState = self.getState(self.env.sense(self))  # retrieves the next state -> inputs & next_waypoint
        nextQ = max([self.getQ(nextState, nextAction) for nextAction in self.actions])  # retrieves max q-value
        # update q_table with new learnings
        self.q_table[(self.state, action)] = self.alpha_learning_rate * (reward + self.gamma_future_reward_discount * nextQ) + (1 - self.alpha_learning_rate) * thisQ

        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        
        # Statistics - to measure performance
        self.num_moves += 1
        self.num_moves_total += 1
        # self.trial_count += 1
        if reward < 0: #Assign penalty if reward is negative
            self.penalty+= 1
            self.show_penalties.append([reward, inputs, self.next_waypoint, action, self.trial_count])
            self.add_up_penalty_score = self.add_up_penalty_score + reward
            self.group_penalties_per_trial.append([self.trial_count, 1])

        add_total = False
        if deadline == 0:
            add_total = True
        if reward >= 10: #agent has reached destination 
            self.reach_dest += 1
            add_total = True
        if add_total:
            self.penalties_to_rewards_ratio = abs(self.add_up_penalty_score) / self.reward_total_100trials * 100
            # self.penalties_to_rewards_progression.append([self.trial_count, ("{0:.2f}".format(round(self.penalties_to_rewards_ratio,2)))])
            self.penalties_to_rewards_progression.append([self.trial_count, self.penalties_to_rewards_ratio])
            self.group_number_of_moves_per_trial.append([self.trial_count, self.num_moves])
            
            print "# of moves:", self.num_moves
            print "# of penalties:", self.penalty
            print "reward total:", self.rewardTotalForTrip
            print "destination reached:", self.reach_dest, "out of", self.trial_count
            # print "self.show_penalties", self.show_penalties # noticed that the agent runs red lights all the time (disobey's traffic law)
            # print "self.penalties_to_rewards_progression:",  self.penalties_to_rewards_progression
            print "self.add_up_penalty_score:", self.add_up_penalty_score
            print "self.num_moves_total:", self.num_moves_total
            print "self.reward_total_100trials:", self.reward_total_100trials
            print "self.penalties_to_rewards_ratio:", ("{0:.2f}".format(round(self.penalties_to_rewards_ratio,2))),"%"
            print "....."
            
        if self.trial_count == 100:
        	if self.run_graph_once == 0:
        		# show_data_views(self.penalties_to_rewards_progression, self.group_penalties_per_trial, self.group_number_of_moves_per_trial)
        		# show_graph(self.penalties_to_rewards_progression, self.group_penalties_per_trial, self.group_number_of_moves_per_trial)
        		self.run_graph_once = 1
               	
    def getState(self, inputs):
        # Get info on inputs and next_waypoint.
        # returns tuple of actions.  # tuple: finite ordered list of elements
        return tuple([tuple(inputs.values()), self.next_waypoint])

    def getAction(self, state):
        # Policy that processes the state of the agent and environment
        # Returns an action such as 'forward', 'left', etc...
        qs = [self.getQ(state, action) for action in self.actions]
        maxQ = max(qs)
        idx = random.choice([i for i in range(len(self.actions)) if qs[i] == maxQ])
        return self.actions[idx]
              
    def getQ(self, state, action):
        # Calculates the Q-value using the state and action variables
        # Returns Q-value
        if (state, action) not in self.q_table:
            return 0
        return self.q_table[(state, action)]
   
def show_data_views(penalties_to_rewards_progression, show_penalties_per_trial, group_number_of_moves_per_trial):
	
	##### View data ####################################
	#pandas_dataframe = pd.DataFrame(boston.data)
	#pandas_dataframe.columns = boston.feature_names
	#pandas_dataframe['Price'] = boston.target
	#print pandas_dataframe.head()
	#print pandas_dataframe.query('Price > 20 & Price < 22')
	#print pandas_dataframe.query('RM > 8')
	#print pandas_dataframe
	##### View data - END ##############################
	
	trial_number = [x[0] for x in penalties_to_rewards_progression]
	progressiveRatio_penalties_rewards = [x[1] for x in penalties_to_rewards_progression]
	
	penalties_grouped_by_trial_number = list(accumulate(show_penalties_per_trial))
	trial_numberB = [x[0] for x in penalties_grouped_by_trial_number]
	grouped_penalty_total_by_trial = [x[1] for x in penalties_grouped_by_trial_number]
	
	group_number_of_moves_per_trial_ordered = list(accumulate(group_number_of_moves_per_trial))
	trial_numberC = [x[0] for x in group_number_of_moves_per_trial_ordered]
	grouped_number_of_moves_total_by_trial = [x[1] for x in group_number_of_moves_per_trial_ordered]
	
	# print "printing from panda function"
	# print "penalties_to_rewards_progression:", penalties_to_rewards_progression
	# print "show_penalties_per_trial", show_penalties_per_trial
	# print "group_number_of_moves_per_trial", group_number_of_moves_per_trial 
	
	pandas_dataframe1 = pd.penalties_to_rewards_progression
	pandas_dataframe1.columns = "trial", "percentage"
	pandas_dataframe1.data = trial_number, progressiveRatio_penalties_rewards
	print "pandas_dataframe1.head():", pandas_dataframe1.head()


def accumulate(l):
    it = itertools.groupby(l, operator.itemgetter(0))
    for key, subiter in it:
       yield key, sum(item[1] for item in subiter)

def show_graph(penalties_to_rewards_progression, show_penalties_per_trial, group_number_of_moves_per_trial):

	trial_number = [x[0] for x in penalties_to_rewards_progression]
	progressiveRatio_penalties_rewards = [x[1] for x in penalties_to_rewards_progression]
	
	penalties_grouped_by_trial_number = list(accumulate(show_penalties_per_trial))
	trial_numberB = [x[0] for x in penalties_grouped_by_trial_number]
	grouped_penalty_total_by_trial = [x[1] for x in penalties_grouped_by_trial_number]
	
	group_number_of_moves_per_trial_ordered = list(accumulate(group_number_of_moves_per_trial))
	trial_numberC = [x[0] for x in group_number_of_moves_per_trial_ordered]
	grouped_number_of_moves_total_by_trial = [x[1] for x in group_number_of_moves_per_trial_ordered]
	
	pl.figure()
	pl.title('Title: Penalties - dissected')
	pl.plot(trial_number, progressiveRatio_penalties_rewards, lw=2, label = 'penalty/reward ratio decrease as trials increase')
	pl.plot(trial_numberB, grouped_penalty_total_by_trial, lw=2, label = '# of penalties per trial')
	# pl.plot(trial_numberC, grouped_number_of_moves_total_by_trial, lw=2, label = '# of moves per trial')
	pl.legend()
	pl.xlabel('trial')
	pl.ylabel('penalties')
	pl.show()

def run(n_trials):
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.0001)  # reduce update_delay to speed up simulation
    sim.run(n_trials=n_trials)  # press Esc or close pygame window to quit
    
    show_data_views(self.penalties_to_rewards_progression, self.group_penalties_per_trial, self.group_number_of_moves_per_trial)
	
if __name__ == '__main__':
    run(100)
