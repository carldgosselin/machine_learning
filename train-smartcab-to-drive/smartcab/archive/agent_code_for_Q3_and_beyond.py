import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

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
        self.penalty = 0
        self.reach_dest = 0
        self.trial_count = 0
        self.rewardTotalForTrip = 0
        
        # for Q4/Q5
        self.epsilon_explore_vs_exploit = 0.2		# amount of exploration to do prior to exploitation (exploiting values in Q-table)
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
        #num2 = float(trial_count)/300
        num2 = 1.0/float(n_trials_plus_1 - trial_count)
        decayed_rate = float(current_rate) - float(num2)
        if decayed_rate <= 0:
        	return 0
        else:
        	return decayed_rate

    def alphaRateDecay(self, current_rate, trial_count):
        # Decay the alpha rate as trial runs approach 100
        n_trials_and_then_some = 100 + 50  # increased number as rate decayed too quickly
        #num2 = float(trial_count)/3000
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
        # self.state = (self.next_waypoint, inputs)	######## Code change for Q2 - choose useful state variables ########
        # print "self.state:", self.getState	######## Code change for Q2 - choose useful state variables ########
        self.state = self.getState(inputs)	######## Code change for Q3 & beyond - created new function. returns tuple ordered list ########
        
        # TODO: Select action according to your policy
        # action = None		######## Original code ########
        # action = self.next_waypoint	######## Experiment ########
        # options = ['forward', 'right', 'left', None]	######## Code change for Q1 - driving randomly ########
        # action = random.choice(options)	######## Code change for Q1 - driving randomly ########
        
        # Epsilon-greedy - explore vs exploit
    # if self.trial_count < 100: # last 20 trials done with full exploitation. Just want to know if agent will incur penalties after 100 trials.
        if random.random() < self.epsilon_explore_vs_exploit:
        	action = random.choice(self.actions) # explore
        else:
        	action = self.getAction(self.state) # exploit
    # else: # no more exploration after 100 trials. time to fully exploit the data!
        # action = self.getAction(self.state) # exploit
               
        # Execute action and get reward
        reward = self.env.act(self, action)
        self.rewardTotalForTrip += reward
             
        # TODO: Learn policy based on state, action, reward
        thisQ = self.getQ(self.state, action)  # returns 0 if state/action tuple not yet in q_table
        nextState = self.getState(self.env.sense(self))  # retrieves the next state -> inputs & next_waypoint
        nextQ = max([self.getQ(nextState, nextAction) for nextAction in self.actions])  # retrieves max q-value
        # update q_table with new learnings
        self.q_table[(self.state, action)] = self.alpha_learning_rate * (reward + self.gamma_future_reward_discount * nextQ) + (1 - self.alpha_learning_rate) * thisQ

        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        
        # Statistics - to measure performance
        self.num_moves += 1
        # self.trial_count += 1
        if reward < 0: #Assign penalty if reward is negative
            self.penalty+= 1
        add_total = False
        if deadline == 0:
            add_total = True
        if reward >= 10: #agent has reached destination 
            self.reach_dest += 1
            add_total = True
        if add_total:
            print "# of moves:", self.num_moves
            print "# of penalties:", self.penalty
            print "reward total:", self.rewardTotalForTrip
            print "destination reached:", self.reach_dest, "out of", self.trial_count
            print "....."

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
   
def run(n_trials):
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.0001)  # reduce update_delay to speed up simulation
    sim.run(n_trials=n_trials)  # press Esc or close pygame window to quit
	
if __name__ == '__main__':
    run(100)
