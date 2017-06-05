# distrl
Distributed Reinforcement Learning

## Installation ##
* Install Anaconda
* Create environment `conda create --name py35 python=3.5`
* Activate environment (Windows) `conda activate py35`
* Activate environment (Ubuntu)  `source conda activate py35`
* Install Python packages

```
conda install -y numpy
conda install -y scipy
pip install tensorflow
pip install gym
pip install baselines
```

## Goal for this project ##
Implement a distributed DQN algorithm according to [this specification](https://docs.google.com/document/d/1e-qyvwNSRq7npmq5qqD5Il9vW4LCe7U6FWBzS1ZLFUw)

## Files to edit ##
The custom Cart Pole script is located at

    baselines/deepq/experiments/custom_cartpole.py
    
Most edits will be done there and in the build graph file at

    baselines/deepq/build_graph.py

