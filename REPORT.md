[//]: # (Image References)

[image1]: img/untrained-gif.gif "Untrained Agent"
[image2]: img/trained-gif.gif "Trained Agent"
[image3]: img/first-score.png "First Score"
[image4]: img/maddpg.JPG "Multi-agent decentralized actor, centralized critic"
[image5]: img/maddpg-algo.png "MADDPG Pseudo-Code"
[image6]: img/final-score.png "Final Score"


# Project 3: Collaboration and Competition

## The Environment

For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Untrained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

## Solution - Multi-Agent DDPG

The implementation follows the Multi-Agent Actor Critic approach presented in this [paper](https://proceedings.neurips.cc/paper/2017/file/68a9750337a418a86fe06c1991a1d64c-Paper.pdf).

![Multi-agent decentralized actor, centralized critic][image4]

Following the Multi-Agent Actor Critic approach, each racket is represented by an individual actor network only sampling from experiences observable by the individual agent, while the critic samples from the experiences of both agents.

![MADDPG Pseudo-Code][image5]
[Image Source](https://arxiv.org/pdf/1706.02275.pdf)


### Results 
At first, training turned out to be very slow. Only after ~2000 episodes the average score began to increase until it eventually started alternating around 0.35.

![First Score][image3]

Finally, the environment was solved in x episodes with an verage score of y over the last 100 episodes.

![Trained Agent][image2]

![Final Score][image6]

### Actor Layout

| Layer | Size | Description |
| ------------- | ------------- | ------------- |
| Input  | 24  | State size of a single agent |
| 1st Hidden Layer  | 24x400  | |
| 2nd Hidden Layer  |400x300  | |
| 3rd Hidden Layer  | 300x2 | |
| Output  | 2  | Action size |

### Critic Layout

| Layer | Size | Description |
| ------------- | ------------- | ------------- |
| Input  | 48 | State size of both agents |
| 1st Hidden Layer  | 48x400  | |
| 2nd Hidden Layer  |404x300  | |
| 3rd Hidden Layer  | 300x1 | |
| Output  | 1  |  |

### Hyperparameters

| Parameter | Value |
| ------------- | ------------- |
| BUFFER_SIZE  | 1e6  |
| BATCH_SIZE  | 256  |
| GAMMA  | 0.99  |
| TAU  | 1e-3  |
| LR_ACTOR  | 1e-4  |
| LR_CRITIC  | 1e-4  |
| WEIGHT_DECAY  | 0  |



## Future Work
In order to further speed up the learning rate we could use priority experience replay. A priority buffer, in addition to the agents' experiences, maintains a sampling probability for ech experience in the buffer which is proportional to its respective training loss (expected result vs actual result). The higher the sampling probability, the greater the chance that an experience is sampled which makes the network learn a lot. 




