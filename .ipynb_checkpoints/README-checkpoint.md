# Simple Self Driving Car using Reinforcement Learning
By Nishat Parveen

To understand how Reinforcement Learning is being used to make autonomous driving possible, it is important to implement the fundamentals in the most basic model of a self driving car.

## Simple_road environment
A very basic scenario, but useful to apply and understand RL concepts.
The envrionment is designed similarly to [openai gym](https://github.com/openai/gym) [`env`](https://gym.openai.com/envs/#classic_control)

### Dependencies

Create a **conda environment**:
`conda create --name rl-for-driving-car python=3.7`

Install the packages:
`pip install -r requirements.txt`

#### Example of settings

After training, env_configuration.json is generated to **summarize the configuration**.

```
{
  "min_velocity":0,
  "previous_action":null,
  "initial_state":[
    0,
    3,
    12
  ],
  "max_velocity_2":2,
  "state_ego_velocity":3,
  "obstacle1_coord":[
    12,
    2
  ],
  "actions_list":[
    "no_change",
    "speed_up",
    "speed_up_up",
    "slow_down",
    "slow_down_down"
  ],
  "goal_velocity":3,
  "goal_coord":[
    19,
    1
  ],
  "previous_state_position":0,
  "obstacle":null,
  "initial_position":[
    0,
    0
  ],
  "previous_state_velocity":3,
  "state_features":[
    "position",
    "velocity"
  ],
  "state_obstacle_position":12,
  "obstacle2_coord":[
    1,
    3
  ],
  "rewards_dict":{
    "goal_with_bad_velocity":-40,
    "negative_speed":-15,
    "under_speed":-15,
    "action_change":-2,
    "over_speed":-10,
    "over_speed_near_pedestrian":-40,
    "over_speed_2":-10,
    "per_step_cost":-3,
    "goal_with_good_velocity":40
  },
  "max_velocity_1":4,
  "max_velocity_pedestrian":2,
  "using_tkinter":false,
  "state_ego_position":0,
  "reward":0
}
```
