#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 21:27:23 2024

@author: ni-kuanshen

Problem:Vaccine Development with Dynamic Programming

You are the CEO of a biotech company which is considering the development of a new vaccine. 
Starting at phase 0 (state 0), the drug develpment can stay in the same state or 
advance to "phase 1 with promising results" (state 1) or advance to "phase 1 with disappointing results" (state 2), 
or fail completely (state 4). 
At phase 1, the drug can stay in the same state, fail or become a success (state 3), 
in which case you will sell its patent to a big pharma company for $10 million. 
These state transitions happen from month to month, and at each state, 
you have the option to make an additional investment of $100,000, which increases the chances of success.
After careful study, your analysts develop the program below to simulate different scenarios using statistical data from similar projects.
Use a discount factor of 0.996.
Write a policy iteration or value iteration code to compute the value of this project. 
Please print the full V (value function) vector.

"""

#%% Given Problem 
'''
Simulating a Markov Decision Process (MDP).

5 states, self.S
0: Phase 0
1: Phase 1 with promising results
2: Phase 1 with disappointing results
3: Success (sell the vaccine patent for $10M)
4: Failure

2 Actions, self.A

0 (no additional investment) 
1 (additional investment) 

P0 and R0: Probabilities and rewards for no investment.
P1 and R1: Probabilities and rewards for additional investment.

Step function: See what happens in 1 step
Calulates and retrieves the next state and reward
If you’ve reach state 4(failure), the simulation ends.

Simulate Function: 
This simulates the entire vaccine development process until it ends:
1. Start in an initial state s and action a.
2. Follow a given policy π to decide the next action at each step.
3. Track the history of states, actions, and rewards.
4. Stop when the process reaches the terminal state (done = True).


'''

#%% Given

import numpy as np
class MDP():
  def __init__(self):
    self.A = [0, 1] 
    self.S = [0, 1, 2, 3, 4]

    P0 = np.array([[0.5, .15, .15, 0, .20],
                   [0, .5, .0, .25, .25],
                   [0, 0, .15, .05, .8],
                   [0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 1]])

    R0 = np.array([0, 0, 0, 10, 0])

    P1 = np.array([[0.5, .25, .15, 0, .10],
                   [0, .5, .0, .35, .15],
                   [0, 0, .20, .05, .75],
                   [0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 1]])

    R1 = np.array([-0.1, -0.1, -0.1, 10, 0])

    self.P = [P0, P1] # Transition probabilities for each action
    self.R = [R0, R1] # Rewards for each action

  def step(self, s, a):
    s_prime = np.random.choice(len(self.S), p=self.P[a][s])
    R = self.R[a][s]
    if s_prime == 4:
      done = True
    else:
      done = False
    return s_prime, R, done

  def simulate(self, s, a, π):
    done = False
    t = 0
    history = []
    while not done:
      if t > 0:
        a = π[s]
      s_prime, R, done = self.step(s, a)
      history.append((s, a, R))
      s = s_prime
      t += 1

    return history
#%% New

# Policy Iteration function
def policy_iteration(mdp, gamma=0.996, tol=1e-6):
    num_states = len(mdp.S)  # Number of states
    num_actions = len(mdp.A)  # Number of actions
    
    # Initialize policy and Value Function
    policy = np.zeros(num_states, dtype=int)
    V = np.zeros(num_states)  
    
    while True:
        # Policy Evaluation
        while True:
            V_new = np.zeros(num_states)
            for s in range(num_states):
                a = policy[s]  
                # Compute the value for state s using the Bellman equation
                V_new[s] = mdp.R[a][s] + gamma * np.dot(mdp.P[a][s], V)
            # Check if the value function has converged
            if np.max(np.abs(V_new - V)) < tol:
                break
            V = V_new
            '''
The goal of this loop is to compute the value function for a given policy
by repeatedly solving the Bellman equation until the value function converges. 
This ensures the value function is accurate, stable, and satisfies the Bellman 
equation for the given policy.       
            '''

        # Policy Improvement
        policy_stable = True
        for s in range(num_states):
            old_action = policy[s]  
            action_values = np.zeros(num_actions)
            # Compute the value for each action in state s
            for a in range(num_actions):
                action_values[a] = mdp.R[a][s] + gamma * np.dot(mdp.P[a][s], V)
            # Update the policy to choose the action with the highest value
            policy[s] = np.argmax(action_values)
            if policy[s] != old_action:
                policy_stable = False  # Policy has changed
            '''
The goal of this loop is to update the policy by finding the best action for each state
based on the current value function. For each state, the algorithm calculates the expected
reward for all possible actions (using the Bellman equation) and selects the action with
the highest value.

If the action for any state changes, the algorithm flags the policy as unstable 
(policy_stable = False) and reverts to Policy Evaluation to recompute the value function 
for the updated policy. This process ensures the policy improves iteratively until it
becomes stable, guaranteeing that the final policy is optimal.

            '''

        # If the policy is stable, the iteration is complete
        if policy_stable:
            break
    
    return V, policy  # Return the final value function and optimal policy

# Create an MDP instance
mdp = MDP()

# Perform policy iteration
V, optimal_policy = policy_iteration(mdp)

# Print the results
print("Value Function (V):", V)
print("Optimal Policy (π):", optimal_policy)

'''
The value function indicates the maximum reward 
that can be achieved from each state,
The optimal policy provides the corresponding actions to achieve those rewards. 
Starting from state 0, the project is worth $3.32 million in expected discounted rewards, 
and the optimal decision is to invest.
'''



















