## Description

### Comparative Analysis of DDPG and PPO Agent Performance Using Distinct Reward Structures

This code evaluates the performance of two reinforcement learning agents, DDPG and PPO, each subjected to two different rewards.
Example codes from SKRL were used to ensure consistent implementation and comparison across both scenarios.


### Install requirements

```
pip3 install -r requirements.txt
```

-------------------------------------------

### Code structure

```
code
├── DDPG.py
├── PPO.py
├── reward.py
└── images
    ├── reward.png
    └── total_reward_mean.png
README.md
requirements.txt
```


-------------------------------------------



### Visualize Reward

```
python3 reward.py
```

<figure>
  <img src="code/images/reward.png" alt="Reward analysis" width="50%">
</figure>


### Run DDPG

1) Sparse Reward

```
python3 DDPG.py --nb_timesteps 50000 --reward_type sparse
```

2) Dense Reward

```
python3 DDPG.py --nb_timesteps 50000 --reward_type dense
```

-------------------------------------------


### Run PPO

1) Sparse Reward

```
python3 PPO.py --nb_timesteps 50000 --reward_type sparse
```

2) Dense Reward

```
python3 PPO.py --nb_timesteps 50000 --reward_type dense
```

### Results

<figure>
  <img src="code/images/total_reward_mean.png" alt="Total Reward" width="100%">
</figure>