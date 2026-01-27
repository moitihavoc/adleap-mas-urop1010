import numpy as np
import datetime

class Waypoints:

    #TODO docstrings and tests

    def __init__(self, angle_discriminant=10.0, min_point_threshold=100):
        """
        Creates a PatternHandler.

        :param float angle_discriminant: Angle in degrees used to compare adjacent points and remove if heading difference is less than it.
        Defaults to 10 deg. Due to small-angle approximation it should be below 0.8 rad (35 deg) to ensure a reasonable level of accuracy.
        :return: An instance of PatternHandler
        :rtype: PatternHandler
        """
        self.points = []
        self.smoothedpoints = None
        angle_disc_rads = angle_discriminant * np.pi / 180.0
        self.angle_discriminant = angle_disc_rads*angle_disc_rads #squared bc small-angle approximation (SAA)
        self.min_point_threshold = min_point_threshold
        self.time_since_obs = 0
        self.is_waypoints_estimated = False

    def reset_handler(self):
        self.points = []
        self.smoothedpoints = None
        self.is_waypoints_estimated = False

    def new_observation(self, boids):
        if (self.time_since_obs%10)==0: #TODO magic number
            self.points.append((boids.sum(0)/len(boids))[0])
        self.time_since_obs+=1
        if len(self.points) > self.min_point_threshold:
            self.is_waypoints_estimated = True

    def get_learned_waypoints(self, force_update=False):
        if self.smoothedpoints is not None and self.is_waypoints_estimated and not force_update:
            return self.smoothedpoints
        return_data = []
        if len(self.points) < self.min_point_threshold:
            return np.array([])

        for i, point in enumerate(self.points):
            if i < 3:
                continue
            buffer=self.points[i - 3 : i] #i.e. a window of 3 points
            vector_a = buffer[1]-buffer[0]
            vector_a /= np.sqrt(vector_a[0]*vector_a[0] + vector_a[1]*vector_a[1])
            vector_b = point-buffer[1]
            vector_b /= np.sqrt(vector_b[0]*vector_b[0] + vector_b[1]*vector_b[1])
            if (2-2*np.dot(vector_a,vector_b)) > self.angle_discriminant: #SAA: 2-2cosT \approx T^2
                return_data.append(point)

        self.smoothedpoints = np.array(return_data)
        return return_data
