# AIM
Experiments for Adversarial Intrinsic Motivation (AIM) on various grid world domains.

## Installation
The code needs python 3.5, and the following packages:
* numpy
* pytorch version 1.6
* matplotlib

## Instructions
* Run main.py: `python main.py`
* Different reward functions can be passed as an argument: `python main.py --reward aim`
* Other available rewards are ['gail', 'airl', 'fairl', 'aim', 'none'], where 'none' uses the sparse task reward
* You can specify the directory where the results are saved using the argument `--dir`,
 for example: `python main.py --reward aim --dir aim_results`
