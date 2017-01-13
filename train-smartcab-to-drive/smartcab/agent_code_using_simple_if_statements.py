import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        
        # TODO: Initialize any additional variables here
        # Statistics - to measure performance
        self.num_moves = 0
        self.penalty = 0
        self.reach_dest = 0
        self.trial_count = 0
        self.rewardTotalForTrip = 0
        
        # for Q4/Q5
        #self.epsilon = 0.1		# amount of exploration to do prior to exploitation (exploiting values in Q-table)
        #self.alpha = 0.3		# learning rate
        #self.gamma = 1			# discount for future awards


    def reset(self, destination=None):
        self.planner.route_to(destination)
        
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.num_moves = 0
        self.penalty = 0
        self.rewardTotalForTrip = 0
        self.trial_count += 1

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = (self.next_waypoint, inputs)		######## Code change for Q2 - choose useful state variables ########
        # print "self.state:", self.state				######## Code change for Q2 - choose useful state variables ########
        
        # TODO: Select action according to your policy
        # action = None									######## Original code 	########
        # action = self.next_waypoint					######## Experiment 	########
        # options = ['forward', 'right', 'left', None]	######## Code change for Q1 - driving randomly ########
        # action = random.choice(options) 				######## Code change for Q1 - driving randomly ########
        
        ########################################################################################################
        ######## Manual code for comparison only (is Q-Learning better than a series of if statements?) ########
        
        ## Series of if statements ##
        action_options = [None, None]
        if(self.state[1]['light'] == 'red'):
        	if(self.state[1]['oncoming'] != 'left'):
        		action_options = ['right', None]
                if self.state[0] == action_options[0]: # right == right
                	action = action_options[0]		# turn right
                else:
                	action = action_options[1]		# wait at intersection. reward = 1
                	# action = action_options[0]	# turn right instead of waiting. reward = 0.5
        else:
            # traffic light is green and now check for oncoming
            if(self.state[1]['oncoming'] == 'forward'):
            	action_options = ['forward','right']
            	if self.state[0] == action_options[0]:  # forward == forward
                	action = action_options[0]		# go forward
                else:
                	action = action_options[1]		# turn right
            else:  # no oncoming traffic 
                action_options = ['right','forward','left']
                if self.state[0] == action_options[0]:  # right == right
                	action = action_options[0]		# turn right
                elif self.state[0] == action_options[1]:  # forward == forward
                	action = action_options[1]		# go forward
                elif self.state[0] == action_options[2]:  # left == left
                	action = action_options[2]		# turn left

        ######## Manual code for comparison only (is Q-Learning better than a series of if statements?) ########
        ########################################################################################################
        
        # Execute action and get reward
        reward = self.env.act(self, action)
        self.rewardTotalForTrip += reward
        
        # TODO: Learn policy based on state, action, reward

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
