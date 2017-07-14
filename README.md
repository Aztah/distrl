# distrl
Distributed Reinforcement Learning

## Installation ##
* Install Anaconda
* Create environment `conda create --name py36 python=3.6`
* Activate environment (Windows) `conda activate py36`
* Activate environment (Ubuntu)  `source conda activate py36`
* Install Python packages

```
pip install tensorflow
pip install gym
pip install dill
(conda install -y scipy)
```

## Goal for this project ##
Implement a distributed DQN algorithm according to [this specification](https://docs.google.com/document/d/1e-qyvwNSRq7npmq5qqD5Il9vW4LCe7U6FWBzS1ZLFUw)

## How to run ##

    cd baselines/deepq/experiments/
    python async_fed_avg.py [--config_file config.ini]
                            [--config DEFAULT]
                             --job_name "worker"
                             --task_index 0
                            [--seed 1]

## Files to edit ##
The asynchronous Cart Pole script is located at

    baselines/deepq/experiments/async_fed_avg.py
    
Most edits will be done there and in the build graph file at

    baselines/deepq/build_graph.py
