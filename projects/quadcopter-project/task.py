import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        sim_position = self.sim.pose[:3]
        euler_angles = self.sim.pose[3:]
        distance = np.sqrt(((sim_position - self.target_pos)**2).sum())
        
        # Base reward amount
        reward = 300.

        # Calculate penalties
        penalty = 0
        #angles_total = abs(euler_angles).sum()
        # Discourage large pitches and rolls
        #penalty = 2. * abs(euler_angles[0:1]+euler_angles[2:]).sum()
        #penalty += 3. * abs(self.sim.linear_accel).sum()

        #if(sim_position[0] < -100 or sim_position[0] > 120 or sim_position[1] < -100 or sim_position[1] > 120 or sim_position[2] > 200):
        #    penalty += 10000

        base_reward_per_unit = 100
        distance_closed = 173 - distance

        if distance > 173:
            reward -= 0
        elif distance > 10 and distance <= 173:
            reward +=  distance_closed*base_reward_per_unit #17000. / (distance)
        elif distance <= 10:
            reward += 17300.
            
        return reward - penalty

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state