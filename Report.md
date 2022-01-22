[//]: # (Image References)

[image1]: ./assets/dqn_results_128.png "DQN result 128"
[image2]: ./assets/dqn_results_64.png "DQN result 64"
[image3]: ./assets/double_dqn_results_128.png "Double DQN result 128"
[image4]: ./assets/double_dqn_results_64.png "Double DQN result 64"
[image10]: ./assets/state_value_fn.png "State value function"
[image11]: ./assets/action_value_fn.png "Action value function"
[image12]: ./assets/bellman_equation.png "Bellman Optimality equation"
[image13]: ./assets/dqn_update.png "DQN update"

**The goal of the project :**

The goal of the project is to train an agent to navigate and collect bananas!) in a square world.
The state space has 37 dimensions and the action space has 4 discrete actions (forward, backward, turn left, turn right).
The agent should collect as many yellow bananas as possible while avoiding blue bananas.
In order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.
	
**The learning algorithms :**

In order for the agent to learn quantification of the amount of reward an action or state
would return is needed and thus value functions can be used; State-Value Function and Action-Value Function.
- State-Value Function measures how good is a state S for an agent to be in at time t while following a policy π.

![state_value_fn][image10]

- Action-Value Function measures how good it is for the agent to take any given action a from a given state S while following a policy π.

![action_value_fn][image11]

(Rt is the immediate reward and γ is the discount factor)

In the case of the value-based methods, the optimal policy has an optimal action-value function (called Q-function) denoted as q* which must satisfy the following equation (Bellman Optimality Equation) :

![Bellman Optimality equation][image12]

(S' is the potential next state and a' is the potential next action)

One of the value-based methods, The Q-learning algorithm, does iteratively update the Q-values for each state-action pair using the Bellman equation until the Q-function converges to the optimal Q-function, q∗. Though, Q-learning has some limitations, it only works with a discrete space (discrete state
and/or discrete action), which is the reason Deep Q-Networks (DQN) got introduced.

The Deep Q-Network uses Deep neural networks to estimate the Q-values for each state-action pair in a given environment. The network will approximate the optimal Q-function, and will get updated throughout the following loss function :

![DQN update][image13]

In order for the agent to learn from past experience, we use Experience Replay in which we store the agent’s experiences at each time step ;
	- It will only store the last N experiences.
	- We’ll randomly sample from it to train the network.
	- It will help to break the correlation between consecutive samples.
	
So far, Q-values will be updated with each iteration to move closer to the target Q-values, but the target Q-values will also be moving in the same direction.
We can solve this using the Target network that will clone the local network (weights of the target network will be frozen and will only be updated every certain amount of steps) and will be used to find the target Q-values.

To help reduce the over-estimation of Q-values, a variation of the DQN algorithm can be used; Double DQN.
It consists of two networks to compute Q Target :
	- Local network to select what is the best action to take for the next state.
	- Target network to calculate the target Q value of taking that action at the next state.
	
	
In this project, 2 algorithms will be used :
	- DQN with fixed-targets and experience replay.
	- Double DQN with fixed-targets and experience replay.
	


**DQN Hyperparameters:**

	BUFFER_SIZE = int(1e5)  # replay buffer size
	BATCH_SIZE = 64         # minibatch size
	GAMMA = 0.99            # discount factor
	TAU = 1e-3              # for soft update of target parameters
	LR = 5e-4               # learning rate 
	UPDATE_EVERY = 4        # how often to update the network
	
	
**DQN neural network architecture 1 :**

*   (fc1): Linear(in_features=37, out_features=128, bias=True) + ReLU()
*   (fc2): Linear(in_features=128, out_features=128, bias=True) + ReLU()
*   (fc3): Linear(in_features=128, out_features=4, bias=True) 
*   Loss: MSELoss()

**DQN neural network architecture 1 :**

	*   (fc1): Linear(in_features=37, out_features=64, bias=True) + ReLU()
    	*   (fc2): Linear(in_features=64, out_features=64, bias=True) + ReLU()
    	*   (fc3): Linear(in_features=64, out_features=4, bias=True) 
	*   Loss: MSELoss()

		
**Results:**
	
* **DQN with fixed-targets and experience replay:**
- Neural network architecture 1
- n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995
- Environment solved in episode 421 with an average score of 13.00
- The score reached 15.52 in episode 2000.

![DQN with NN 1 results][image1]

* **DQN with fixed-targets and experience replay:**
- Neural network architecture 2
- n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995
- Environment solved in episode 382 with an average score of 13.04
- The score reached 16.32 in episode 2000.

![DQN with NN 2 results][image2]
		
* **Double DQN with fixed-targets and experience replay:**
- Neural network architecture 1
- n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995
- Environment solved in episode 404 with an average score of 13.01
- The score reached 15.34 in episode 2000.

![Double DQN with NN 1 results][image3]

* **Double DQN with fixed-targets and experience replay:**
- Neural network architecture 1
- n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995
- Environment solved in episode 356 with an average score of 13.04
- The score reached 15.95 in episode 2000.

![Double DQN with NN 2 results][image4]

**Ideas for Future Work:**

- Implement Dueling DQN
- Implement the Prioritized Replay Buffer
