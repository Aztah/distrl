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
    python async_fed_avg.py [--ps_hosts "localhost:13337"]
                            [--worker_hosts "localhost:13338,localhost:13339"]
                            --job_name "worker"
                            --task_index 0
                            [--learning_rate 5e-4]
                            [--batch_size 32]
                            [--memory_size 50000]
                            [--target_update 1000]
                            [--seed 1]
                            [--comm_rounds 500000]
                            [--epochs 100]

or

    python async_fed_avg.py --config <config_file_name>
                            --job_name "worker"
                            --task_index 0

## Files to edit ##
The asynchronous Cart Pole script is located at

    baselines/deepq/experiments/async_fed_avg.py
    
Most edits will be done there and in the build graph file at

    baselines/deepq/build_graph.py
