"""
Sudhanva Sreesha
ssreesha@umich.edu
21-Mar-2018

Gonzalo Ferrer,
g.ferrer@skoltech.ru

Defines the field (a.k.a. map) for this task.
"""

import numpy as np
import rospy

class FieldMap(object):
    def __init__(self, configuration='yellow'):
        ''' Class Field Map defines the configuration of the map 
            and containts ground truth data of the landmark positions

            Args:
                configuration: yellow or blue (depends on which side is active for the robot)

            Returns: Field Map object
        '''

        #if configuration == 'yellow':
            # X poses of the landmarks
        self._landmark_poses_x = self.YELLOW_BEACONS[:, 0]

            # Y poses of the landmarks
        self._landmark_poses_y = self.YELLOW_BEACONS[:, 1]

        #elif configuration == 'blue':
            # X poses of the landmarks
            #self._landmark_poses_x = self.BLUE_BEACONS[:, 0]

            # Y poses of the landmarks
            #self._landmark_poses_y = self.BLUE_BEACONS[:, 1]

    # Define map properties
    @property
    def WORLD_X(self):
        return rospy.get_param("world_x")

    @property
    def WORLD_Y(self):
        return rospy.get_param("world_y")

    @property
    def WORLD_BORDER(self):
        return rospy.get_param("world_border")

    @property
    def BEAC_L(self):
        return rospy.get_param("beac_l")
    
    @property
    def BEAC_BORDER(self):
        return rospy.get_param("beac_border")

    # Define arrays of beacon positions [x, y, id]
    @property
    def YELLOW_BEACONS(self):
        YELLOW_BEACONS = np.array([ [self.WORLD_X + self.WORLD_BORDER + self.BEAC_BORDER + self.BEAC_L / 2., self.WORLD_Y / 2.],
                                    [-(self.WORLD_BORDER + self.BEAC_BORDER + self.BEAC_L / 2.), self.WORLD_Y - self.BEAC_L / 2.],
                                    [-(self.WORLD_BORDER + self.BEAC_BORDER + self.BEAC_L / 2.), self.BEAC_L / 2.]])
        return YELLOW_BEACONS
    
    @property
    def BLUE_BEACONS(self):
        BLUE_BEACONS = np.array([   [-(self.WORLD_BORDER + self.BEAC_BORDER + self.BEAC_L / 2.), self.WORLD_Y / 2.],
                                    [self.WORLD_X + self.WORLD_BORDER + self.BEAC_BORDER + self.BEAC_L / 2., self.WORLD_Y - self.BEAC_L / 2.],
                                    [self.WORLD_X + self.WORLD_BORDER + self.BEAC_BORDER + self.BEAC_L / 2., self.BEAC_L / 2.]])
        return BLUE_BEACONS