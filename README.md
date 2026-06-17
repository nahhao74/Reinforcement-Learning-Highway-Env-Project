# Reinforcement Learning Highway Env Project

## 1. Project Introduction

This project uses the `highway-env` reinforcement learning environment to train an autonomous vehicle agent for highway driving. The objective is to make the vehicle drive efficiently on a highway while avoiding surrounding vehicles and recognizing dangerous driving situations.

The project focuses on building the observation space, designing the reward function, training the reinforcement learning agent, and evaluating the vehicle behavior in a simulated highway environment.

`highway-env` provides simulation environments for autonomous driving and tactical decision-making tasks, making it suitable for testing highway driving policies before real-world deployment.

---

## 2. Engineering Objective

The main objective is to train an ego vehicle that can:

- Keep driving on the highway
- Avoid collisions with surrounding vehicles
- Maintain a suitable driving speed
- Change driving behavior based on nearby traffic
- Recognize dangerous or risky situations
- Improve driving decisions through reinforcement learning

The system is not focused on low-level vehicle dynamics. Instead, it focuses on high-level decision-making for autonomous highway driving.

---

## 3. Engineering Problem

Highway driving requires the vehicle to make safe and efficient decisions in a dynamic environment. The ego vehicle must observe nearby vehicles, evaluate traffic risk, and choose appropriate actions such as maintaining speed, slowing down, or changing lanes.

The main challenge is to design suitable observations and rewards so the agent can learn safe driving behavior. If the reward function is not well designed, the vehicle may drive too slowly, collide with other cars, or make unsafe lane changes.

---

## 4. Methodology

The project workflow is:

```text
Highway driving environment
        ↓
Observation space design
        ↓
Reward function design
        ↓
RL agent training
        ↓
Policy evaluation
        ↓
Highway driving and hazard avoidance
```

---

## 5. Environment Setup

The project is based on the `highway-env` simulation environment.

The environment provides:

- Ego vehicle
- Multiple highway lanes
- Surrounding traffic vehicles
- Driving actions
- Observation data
- Reward feedback
- Collision and safety evaluation

Typical environment elements:

```text
Ego vehicle
Surrounding vehicles
Lane information
Vehicle speed
Relative position
Relative velocity
Collision status
```

---

## 6. Observation Design

The observation is designed to provide the agent with useful information about the highway situation.

The observation data can include:

- Ego vehicle speed
- Ego vehicle lane position
- Relative distance to nearby vehicles
- Relative velocity of nearby vehicles
- Position of surrounding vehicles
- Lane-related information
- Collision or danger-related state

The purpose of the observation space is to help the agent understand whether the current driving situation is safe or dangerous.

---

## 7. Reward Function Design

The reward function guides the agent during training.

The reward design focuses on:

- Positive reward for safe driving
- Positive reward for maintaining a suitable speed
- Penalty for collision
- Penalty for unsafe behavior
- Penalty for risky distance to other vehicles
- Reward for efficient highway driving

A simplified reward objective is:

```text
High reward  -> fast and safe driving
Low reward   -> slow, unsafe, or inefficient driving
Penalty      -> collision or dangerous behavior
```

The reward function is important because it directly affects what behavior the agent learns.

---

## 8. Reinforcement Learning Training

During training, the agent interacts with the highway environment repeatedly.

At each step:

```text
1. The agent receives observation data.
2. The agent selects a driving action.
3. The environment updates the vehicle state.
4. The agent receives a reward.
5. The agent improves its policy through training.
```

The training goal is to learn a driving policy that balances:

- Safety
- Speed
- Obstacle avoidance
- Lane behavior
- Driving efficiency

---

## 9. Hazard Recognition and Avoidance

The trained vehicle is expected to recognize dangerous situations based on observation data.

Examples of dangerous situations:

- Vehicle too close in front
- High relative speed with nearby vehicles
- Risky lane change condition
- Collision risk with surrounding traffic
- Unsafe spacing between vehicles

The agent should respond by selecting safer actions, such as slowing down, keeping lane, or changing lane when the condition is safe.

---

## 10. Expected Result

The expected result is an autonomous highway driving agent that can drive at a suitable speed while avoiding surrounding vehicles.

Key expected behavior:

- The vehicle can keep moving on the highway.
- The vehicle can avoid obstacles and surrounding traffic.
- The vehicle can reduce collision risk.
- The vehicle can react to dangerous situations.
- The vehicle can improve driving behavior after training.

---

## 11. Tools and Technologies

- Python
- highway-env
- Reinforcement Learning
- Autonomous Driving Simulation
- Reward Function Design
- Observation Space Design
- Visual Studio Code

---

## 12. Project Scope

This project focuses on reinforcement learning for high-level highway driving decisions. It does not focus on mechanical vehicle modeling, real vehicle hardware, or low-level actuator control.

The main technical focus is:

- Environment configuration
- Observation design
- Reward function design
- Agent training
- Highway driving behavior evaluation

---

## 13. References

- highway-env GitHub Repository: https://github.com/Farama-Foundation/HighwayEnv
- highway-env Documentation: https://highway-env.farama.org/
