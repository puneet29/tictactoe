# Tic Tac Toe

This repo is an experiment to try making a model to learn how to play tic tac toe with experience. For this I will try using Reinforcement learning using Deep learning.

## Acknowledgement

This is my second try at reinforcement learning. I have not studied reinforcement learning before except that I have tried an OpenAI Gym game before attempting this project. This is solely my own concept and code. In case you find any bugs or discrepencies or a better version, I will be happy to hear. I want to thank @sentdex for his tutorials, the project-first learning approach really does make you learn a lot on the way.

## Experiments

All experiments did before were not accurate because there was a minor logic error in code. I have fixed the code and now everything works fine. All of the models have attained the best possible policy. All of these tie with the minimax(Most Optimal agent). I have ran `300,000` episodes, however each initialization (zero, ones, and random) will have their own learning curves. Also I have only tried training as player 1. The process will be similar for training player 2.

Each models folder contains:

1. `params.txt` which tells the parameters at the time of training,
2. `reward_stats.png` which shows the maximum, average and minimum rewards collected over the buckets of 500 episodes,
3. `results_minimax.txt` which contains the results of the model when competed with minimax agent,
4. `q_visualized.html` (if you run the training process yourself) which contains the visualization of the Q-Table

## Next up

1. **Major refactoring of code is needed**
