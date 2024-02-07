# MultiArmedBandit
# Multi-Armed Bandit Algorithms

## Overview

This repository contains Python code implementing various multi-armed bandit algorithms for reinforcement learning. The algorithms include epsilon-greedy, Upper Confidence Bound (UCB), and their variants. The performance of these algorithms is evaluated using a testbed with synthetic reward distributions.

## Table of Contents

- [Introduction](#multi-armed-bandit-algorithms)
- [Code Structure](#code-structure)
- [Implemented Algorithms](#implemented-algorithms)
- [How to Use](#how-to-use)
- [Results](#results)

## Code Structure

The code is organized into a Python class, which encapsulates the implementation of the multi-armed bandit algorithms. Each algorithm is implemented as a method within this class.

## Implemented Algorithms

The following algorithms are implemented:

1. **Epsilon-Greedy:** A simple exploration-exploitation strategy where the agent chooses the action with the highest estimated value with probability (1 - epsilon) and a random action with probability epsilon.
2. **Upper Confidence Bound (UCB):** An algorithm that balances exploration and exploitation by selecting actions based on their estimated value and uncertainty.
3. **UCB with different exploration parameters:** Variants of UCB with different exploration parameters to study their impact on algorithm performance.

## How to Use

To use the code:

1. Clone the repository to your local machine.
2. Import the `Question1` class from the Python script.
3. Instantiate an object of the `Question1` class with the desired parameters (number of arms, number of runs, etc.).
4. Call the appropriate method to run the desired algorithm and visualize the results.

## Results

The performance of each algorithm is evaluated using synthetic reward distributions generated by the provided testbed. The results are visualized using matplotlib and presented in the form of graphs showing rewards obtained and action selections over time.

