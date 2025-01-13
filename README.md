# Deep Q-Learning (DQN) for CartPole OpenAI Gym
 This project implements Deep Q-Learning (DQN) to solve the CartPole-v1 environment, focusing on the exploration-exploitation trade-off using various exploration strategies (ε-greedy, annealing ε-greedy, and softmax). The project also explores the impact of hyperparameters such as learning rate, batch size, and discount factor on performance, providing insights into optimizing Deep Q-Learning for a standard reinforcement learning problem

# Deep Q-Learning for CartPole: Exploration Strategies, Hyperparameter Tuning, and Performance Evaluation

This project implements **Deep Q-Learning (DQN)** to solve the **CartPole-v1** environment in OpenAI Gym. The main focus is on studying the **exploration-exploitation trade-off** using different exploration strategies, evaluating how **ε-greedy**, **annealing ε-greedy**, and **softmax** strategies affect performance. The project further investigates the impact of **hyperparameter tuning** such as learning rate, batch size, and discount factor on the DQN's learning performance.

## Project Goals
1. **Implement Deep Q-Networks** (DQN) to balance the pole in the **CartPole** environment.
2. **Evaluate different exploration strategies**: ε-greedy, Boltzmann, and annealing ε-greedy.
3. **Hyperparameter Tuning**: Explore the impact of parameters like **learning rate**, **discount factor**, and **batch size** on the performance of DQN.
4. **Performance Evaluation**: Measure the agent's performance using standard RL metrics like **average reward** and **learning curves**.

### Environment Setup:
1. **OpenAI Gym** for simulation and environment management.
2. **TensorFlow** (or PyTorch) for implementing the neural network.
3. **Numpy** for numerical calculations.
4. **Matplotlib** for visualizing learning curves and performance metrics.

### Installation Instructions:
Ensure you have the required libraries before running the code:
1. Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
   ```

2. Install the necessary dependencies using the requirements.txt:

```bash
pip install -r requirements.txt

```
3. To run the DQN algorithm with all default settings, use:
```bash
python DQN.py

```
4. To remove experience replay, use:

```bash
python DQN.py --replay

```

5. To remove the target network, use:

```bash
python DQN.py --target

```

6. To remove both experience replay and target network, use:

```bash
python DQN.py --replay --target

```
7. Check the folder where the code is placed for performance plots. The plot file names will include the hyperparameter configurations.


## Evaluation Results:

| Metric                 | DQN (ε-greedy) | DQN (Annealing ε-greedy) | DQN (Boltzmann) |
|------------------------|----------------|--------------------------|-----------------|
| **Total Steps**         | 475            | 480                      | 460             |
| **Average Loss**        | 0.042          | 0.037                    | 0.040           |
| **Convergence Speed**   | Slow           | Moderate                 | Fast            |

### Training and Evaluation:
The agent is trained over **2000 episodes**. During each episode, the agent interacts with the environment and learns to balance the pole. The **total steps** before the pole drops and the **average loss** are tracked. After training, the agent’s performance is evaluated by measuring the total number of steps the pole is balanced in each episode.

### Results & Discussion:
- **Annealing ε-greedy** outperforms **ε-greedy** and **Boltzmann** policies, showing a balance between exploration and exploitation.
- **Boltzmann** strategy provides smoother learning curves but requires more computational resources due to the softmax calculation.
- **Experience Replay** and **Target Network** improve the agent’s stability and convergence speed.

### Conclusion:
The **Deep Q-Network (DQN)** implementation effectively learns the optimal policy for the **CartPole-v1** environment. The **annealing ε-greedy** strategy provides a good trade-off between exploration and exploitation, with the **Boltzmann** strategy being more stable but computationally expensive.

## References:
1. **OpenAI Gym**: [https://gym.openai.com/envs/CartPole-v1/](https://gym.openai.com/envs/CartPole-v1/)
2. **Deep Q-Learning (Mnih et al., 2015)**: [https://arxiv.org/abs/1312.5602](https://arxiv.org/abs/1312.5602)
3. **Reinforcement Learning: An Introduction (Sutton & Barto, 2018)**: [http://incompleteideas.net/book/the-book-2nd.html](http://incompleteideas.net/book/the-book-2nd.html)
