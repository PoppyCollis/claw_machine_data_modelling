# claw_machine_data_modelling
Modelling claw machine experiment data using Optimal Bayesian decision-making agent.

## Usage

In __main__ section of decision_maker.py:


    - Adjust parameters of the ball distributions via "cats" dictionary
    - Adjust rewards of the ball distributions via "rews" dictionary
    - Provide pairs to evaluate via list of tuples called "pairs"
    - When initialising the agent we need:
        - Threshold
        - beta (inverse temperature) parameter
        - soft: if soft = False, then we do a MAP decision rule, otherwise soft sampling of the catgeory