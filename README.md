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

* `config_file` is the name of the config file (or path if it's in a different directory).
    * Defaults to `config.ini`
    
* `config` is the section in the config file to override the default values with.
    * Defaults to `DEFAULT`. Use `async` for the `[async]` section and `sync` for the `[sync]`

* `job_name` is the type of job the current client should perform.
    * Defaults to `worker`. Use only `worker` or `ps` for this value

* `task_index` is the index of the current server's IP in the list for its job (ps or worker).
    * Defaults to `0`. Worker 0 will become the chief with extra responsibilities

* `seed` is the seed for all the randomness in the server.
    * Defaults to `1`. The server's task index will be added to this to make sure every server has a unique seed

## Files to edit ##
The asynchronous Cart Pole script is located at

    baselines/deepq/experiments/async_fed_avg.py
    
Most edits will be done there and in the build graph file at

    baselines/deepq/build_graph.py
