# PlayConf

PlayConf is the first idea we have for adversarially teaching models how physics looks
like. The main idea is that we have two models, the adversary `M_a` and the predictor
`M_p`. The adversary generates initial environment configurations in order to maximize
the MSE of the predictor and the predictor uses these trajectories to improve itself.

Since the adversary's 'labels' are generated through an environment, we optimize it
using an RL optimizer such as PPO or TRPO. The predictor however can be learned
directly with SGD.

## Environment

The environment loads a physics simulation, and a predictor model. When given an initial
configuration, it runs a few steps through the physics simulation along with the
predictor and computes the net mean squared error of the predictor as reward.

In a traditional RL setup, the adversary's configuration is the action space and the
MSE is the reward.
