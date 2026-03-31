# Repository for the Agents Who Understand When To Plan Work

## High level idea
We want to create agents with a few properties:
1. They understand time cost
    - Creating a plan will take time
    - They won't have that plan instantly
    - They need to do something while they wait
    - They can't just sit there (depending on domain)
2. They understand how planning works (at a high level)
    - They can usually follow a plan, but may need to recreate one
    - When to recreate one?

## Current status

Curriculum learning seems to work well. However, we are in a situation in which the agent gets scared to approach near lava. The high level idea is we want to be able to truly trust plans when they are good, and not trust when they are bad. Ideas:
- We give a small reward for path following
- We add in part of the loss which disourages not staying close to plan following policy for fine tuning

Other ideas:
- Small penalty for calling plan? Avoids getting stuck calling plans
 

The "easy" version works quite well. However, as the tasks get more stochastic and complex, the policy struggles to learn from scratch. 
Ideas: Curriculum learning/fine tuning




Possibly a good domain: [https://github.com/waymo-research/waymax]

## Useful commands

`python when_can_i_follow/train.py envs=basic`