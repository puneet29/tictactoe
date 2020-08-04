# Tic Tac Toe

This repo is an experiment to try making a model to learn how to play tic tac toe with experience. For this I will try using Reinforcement learning using Deep learning.

## Experiments

I am new to reinforcement learning and have not tackled any dual player games yet. This is my first attempt on taking on a dual player game coming straight from the single player game. Also, this is my very second reinforcement learning problem that I am trying to solve and without any help from internet.

I have tried changing the learning rates, reward, no. of episodes, epsilon decay, q-table initializations, player 2 (self, random agent, random steps, trained agent [Only q learners]) but wasn't able to get human like good performance in the game.

On visualizing the q tables, I found out that most of the states were not explored and as our second agent was mostly random, the agent didn't learn much better moves. As I can't play as second player for more than 100 games(being much generous in terms of training episodes). So, for that I have implemented a Minimax Agent to play the game as second player. I hope this will improve our agent way more than a random player. The curiousity is whether it will be as good as the minimax or poor than that.

I have also linked how the models performed upon initialization of q table in each folder with `params.txt` defining all the params at the time of training.

## Next up

1. Training with minimax agent as player 2
2. **Major refactoring of code is needed**
