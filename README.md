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
The "easy" version works quite well. However, as the tasks get more stochastic and complex, the policy struggles to learn from scratch. 
Ideas: Curriculum learning/fine tuning