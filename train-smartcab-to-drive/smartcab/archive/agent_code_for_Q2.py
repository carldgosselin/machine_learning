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

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = (self.next_waypoint, inputs)		######## Code change for Q2 ########
        print "self.state:", self.state					######## Code change for Q2 ########
        
        # TODO: Select action according to your policy
        # action = None
        # action = self.next_waypoint
        options = ['forward', 'right', 'left', None]	######## Code change for Q1 ########
        action = random.choice(options) 				######## Code change for Q1 ########
        
        ######## Experimentation ########
        ## Selecting action based on state
        #info_on_light_and_incoming_car = self.env.sense(self)
        
        #action_options = []
        #if(info_on_light_and_incoming_car['light'] == 'red'):
        #    if(info_on_light_and_incoming_car['oncoming'] != 'left'):
        #        action_options = ['right', None]
        #        if self.next_waypoint == action_options[0]:
        #        	action = action_options[0]
        #        else:
        #        	# action = action_options[0] # turn right instead of waiting. reward = 0.5
        #        	action = action_options[1] # reward = 1
        #else:
        #    # traffic light is green and now check for oncoming
        #    if(info_on_light_and_incoming_car['oncoming'] == 'forward'):
        #    	action_options = [ 'forward','right']
        #    	if self.next_waypoint == action_options[0]:
        #        	action = action_options[0]
        #        else:
        #        	action = action_options[1]
        #    else:  # no oncoming traffic 
        #        action_options = ['right','forward', 'left']
        #        if self.next_waypoint == action_options[0]:
        #        	action = action_options[0]
        #        elif self.next_waypoint == action_options[1]:
        #        	action = action_options[1]
        #        elif self.next_waypoint == action_options[2]:
        #        	action = action_options[2]
        #######################################
        
        # Execute action and get reward
        reward = self.env.act(self, action)
        
        # TODO: Learn policy based on state, action, reward

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=1.0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=10)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
