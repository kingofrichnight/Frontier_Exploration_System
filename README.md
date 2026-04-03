# Frontier_Exploration_System
ROS2-based autonomous robot system for frontier exploration, obstacle-aware navigation, and target search &amp; localization.

# 🤖 Autonomous Frontier Exploration, Navigation, and Localization System

## 📌 Overview
This project presents a complete autonomous robotic system developed using **ROS2**, designed to explore unknown environments, navigate safely in the presence of obstacles, and perform search and localization tasks.

The system is implemented and tested in simulation using **TurtleBot3**, **Gazebo**, and **RViz**, integrating key robotics concepts such as mapping, planning, and perception.

---

## 🚀 Key Features

- 🧭 Frontier-based autonomous exploration  
- 🚧 Obstacle-aware navigation using path planning  
- 🔍 Target search and localization  
- ⚙️ Modular ROS2 architecture  
- 🗺️ Occupancy grid mapping and visualization  

---

**📁 Project Structure**
```bash

Frontier_Exploration_System/
│── launch/ # Launch files for simulation and tasks
│── maps/ # Saved occupancy grid maps
│── models/ # Gazebo simulation models
│── params/ # Configuration files
│── rviz/ # RViz visualization configurations
│── src/Tasks/ # Main task implementations
│ ├── task1.py # Frontier exploration
│ ├── task2.py # Navigation with static obstacles
│ ├── task3.py # Search and localization
│ ├── static_obstacles.py
│ └── spawn_objects.py
│── urdf/ # Robot description
│── worlds/ # Gazebo environments
│── CMakeLists.txt
│── package.xml

```
---

## ⚙️ Requirements

- ROS2 (Humble / Foxy)
- Gazebo
- RViz2
- TurtleBot3 packages

---

## 🛠️ Installation & Setup

### 1. Clone the repository
```bash
git clone <your-repo-link>
cd Frontier_Exploration_System
2. Build the workspace
colcon build
source install/setup.bash
3. Launch Simulation
ros2 launch Frontier_Exploration_System turtlebot3_house.launch.py

```
🔹 Task Breakdown
🧭 Task 1: Frontier Exploration

The Frontier Exploration module enables the robot to autonomously explore unknown environments using a frontier-based strategy. The robot detects boundaries between known and unknown regions, selects exploration goals, and navigates iteratively until the environment is fully explored.

▶ Run Task 1
```bash
ros2 run Frontier_Exploration_System task1

```
---

🚧 Task 2: Navigation with Static Obstacles

This task enables safe navigation in environments with static obstacles. The robot uses path planning and costmaps to generate collision-free trajectories and reach target goals efficiently.

▶ Run Task 2
```bash
ros2 run Frontier_Exploration_System task2
```
⭐ Bonus: Enhanced Navigation
```bash
ros2 launch turtlebot3_gazebo navigator.launch.py static_obstacles:=true bonus:=true
```
---

🔍 Task 3: Search and Localization

This module allows the robot to search for and localize objects within the environment. It integrates perception and navigation to detect targets and estimate their positions.

▶ Run Task 3
```bash
ros2 run Frontier_Exploration_System task3
```
▶ Run with Object Spawning
```bash
ros2 launch turtlebot3_gazebo navigator.launch.py spawn_objects:=true
🎯 Project Objective
```

---

**This project demonstrates a complete autonomous robotics pipeline integrating:**

Mapping
Exploration
Navigation
Obstacle avoidance
Localization
📊 Results

**The robot successfully:**

Explores unknown environments
Builds occupancy grid maps
Avoids obstacles
Detects and localizes targets


---

**👨‍💻 Author**

Aarya F
MS in Autonomy – Purdue University

---

