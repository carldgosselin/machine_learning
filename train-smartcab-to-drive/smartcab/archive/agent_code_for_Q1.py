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
        
        # TODO: Select action according to your policy
        # action = None
        # action = self.next_waypoint
        options = ['forward', 'right', 'left', None]	######## Code change for Q1 ########
        action = random.choice(options) 				######## Code change for Q1 ########
       
        # Execute action and get reward
        reward = self.env.act(self, action)

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

def run(n_trial):
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.0001)  # reduce update_delay to speed up simulation
    sim.run(n_trials=n_trial)  # press Esc or close pygame window to quit

if __name__ == '__main__':
    run(100)
