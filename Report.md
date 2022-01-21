[//]: # (Image References)

[image1]: ./assets/dqn_results_128.png "DQN result 128"
[image2]: ./assets/dqn_results_64.png "DQN result 64"
[image3]: ./assets/double_dqn_results_128.png "Double DQN result 128"
[image4]: ./assets/double_dqn_results_64.png "Double DQN result 64"

**The goal of the project :**

    The goal of the project is to train an agent to navigate and collect bananas!) in a square world.
    The state space has 37 dimensions and the action space has 4 discrete actions (forward, backward, turn left, turn right).
    The agent should collect as many yellow bananas as possible while avoiding blue bananas.
    In order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.
	
**The features :**

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