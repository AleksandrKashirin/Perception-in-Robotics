"""
Gonzalo Ferrer
g.ferrer@skoltech.ru
28-Feb-2021
"""

from tkinter import W
import numpy as np
from tools.jacobian import state_jacobian
import mrob
from scipy.linalg import inv
from slam.slamBase import SlamBase
from tools.task import get_motion_noise_covariance

class Sam(SlamBase):
    def __init__(self, initial_state, alphas, state_dim=3, obs_dim=2, landmark_dim=2, action_dim=3, *args, **kwargs):
        super(Sam, self).__init__(*args, **kwargs)
        self.state_dim = state_dim
        self.landmark_dim = landmark_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.alphas = alphas
        
        self.graph = mrob.FGraph()
        self.Nodes = []   # List of nodes
        self.lm_dict = {} # Dictionary with observed landmarks {observation:lm}

        # Task 1. A: Begin
        # Assign initial state
        self.X_0 = initial_state.mu
        # Get initial node
        self.orig_node_id = self.graph.add_node_pose_2d(self.X_0)
        # Append initial node to the list of nodes
        self.Nodes.append(self.orig_node_id)
        # Set initial information matrix
        self.init_inf_matrix = inv(initial_state.Sigma)
        # Add factor pose
        self.graph.add_factor_1pose_2d(self.X_0, self.orig_node_id, self.init_inf_matrix)
        self.graph.print(True)
        # Task 1. A: End

    def predict(self, u):
        # Task 1. B: Begin
        # Add target node
        target_node_id = self.graph.add_node_pose_2d(np.zeros(3))
        #print('\n X_est before =', self.graph.get_estimated_state())
        # Get the last estimated state
        self.mu = self.graph.get_estimated_state()[self.orig_node_id]
        # Calculate Jacobian with respect to input signal
        _, V = state_jacobian(self.mu.T[0], u)
        # Calculate information matrix
        W_u = inv(V @ get_motion_noise_covariance(u, self.alphas) @ V.T)
        # Add factor to the graph associated with odometry model
        self.graph.add_factor_2poses_2d_odom(u, self.orig_node_id, target_node_id, W_u)
        #print('\n X_est after =', self.graph.get_estimated_state())
        # Reassign the original node
        self.orig_node_id = target_node_id
        self.Nodes.append(self.orig_node_id)
        # Task 1. B: End

    def update(self, z):
        # Task 1. C: Begin
        self.obsv_id = z[:, 2].T # Vector of observed landmarks
        self.W_z = inv(self.Q)   # Information matrix on observation covariance
        # For each obtained observation
        for i in range(len(self.obsv_id)):
            # If landmark was already observed
            if self.lm_dict.get(int(self.obsv_id[i])):
                # Get node landmark id
                self.node_lm_id = self.lm_dict[self.obsv_id[i]]
                # Add factor to the graph assigned with observation
                self.graph.add_factor_1pose_1landmark_2d(z[i,:2], self.orig_node_id, self.node_lm_id, self.W_z, initializeLandmark=False)

            # Else landmark was not previously observed
            else:
                # Create a new landmark node and connect it with the new state
                self.node_lm_id = self.graph.add_node_landmark_2d(np.zeros(2)) 
                # Assign new landmark with observations
                self.lm_dict[self.obsv_id[i]] = self.node_lm_id
                # Add factor to the graph assigned with observation
                self.graph.add_factor_1pose_1landmark_2d(z[i,:2], self.orig_node_id, self.node_lm_id, self.W_z, initializeLandmark=True)
        
    def info(self):
        # For each state node
        for i in range(len(self.Nodes)):
            # Print state node id and estimated state of the pose
            print(f'State Node ID:', self.Nodes[i], ', Estimated state: ', self.graph.get_estimated_state()[self.Nodes[i]].T)

        # For each landmark node
        for j in self.lm_dict.values():
            # Print landmark node id and estimated state of the landmark
            print(f'Landmark node ID:', j, ', Estimated landmark state:', self.graph.get_estimated_state()[j].T)
        # Task 1. C: End

    # Task 1. D: Begin
    def solve(self, method=mrob.GN):
        self.graph.solve(method)

    def graph_print(self):
        self.graph.print(True)
    # Task 1. D: End

    # Task 2. B: Begin
    def get_states(self):
        self.robot_states = []
        self.lm_states = []
        
        # For each state node
        for i in range(len(self.Nodes)):
            # Append estimated state of the pose
            self.robot_states.append(self.graph.get_estimated_state()[self.Nodes[i]].T[0])

        # For each landmark node
        for j in self.lm_dict.values():
            # Append estimated state of the landmark
            self.lm_states.append(self.graph.get_estimated_state()[j].T[0])
        
        return np.array(self.robot_states), np.array(self.lm_states)
    # Task 2. B: End