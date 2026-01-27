import numpy as np
from src.envs.InfiltrationEnv import MAX_FORCE, MAX_FORCE_SQ, MAX_STEER, MAX_STEER_ROT_CCW, MAX_STEER_ROT_CW, MAX_STEER_SQ, BoidParams, ForceTypes, ENTITY_SIZE_SQ, MAX_SPEED
from scipy.ndimage import rotate

def boid_planning(env, agent):
    smart_params = agent.smart_parameters
    boids_state = smart_params["boid"]
    if boids_state is None:
        smart_params["boid"] = Boid(agent.flock_state, smart_params["waypoints"], smart_params["waypoint_idx"], smart_params["boid_params"])
        boids_state = smart_params["boid"]
    else:
        smart_params["waypoint_idx"] = boids_state.update_state(agent.flock_state, smart_params["boid_params"])
    
    acc = boids_state.get_alignment()
    acc += boids_state.get_cohesion()
    acc += boids_state.get_separation()
    acc += boids_state.get_attraction()
    return acc, None


class Boid():
    def __init__(self, flock_state, waypoints, waypoint_idx=1, boid_params=BoidParams.get_default()):

        # agent parameters
        self.flock_state = flock_state
        self.waypoints = waypoints #ndarray of 2d ndarray 
        self.waypoint_idx = waypoint_idx #int
        self.boid_params = boid_params
        dim = self.flock_state.shape
        self.reached_waypoint = np.zeros(dim[1], dtype=bool)
        self.waypoint_threshold = 0.75
        self.align_neighbour_mask = np.zeros((dim[1], dim[1]), dtype=bool)
        self.cohesion_neighbour_mask = np.zeros((dim[1], dim[1]), dtype=bool)
        self.separation_neighbour_mask = np.zeros((dim[1], dim[1]), dtype=bool)
        self.sq_dist = np.zeros((dim[1], dim[1]))
        self._update_neighbour_masks()

    @property
    def current_waypoint(self):
        return self.waypoints[self.waypoint_idx] if self.waypoints is not None else self.center_of_mass

    @property
    def center_of_mass(self):
        # calculates center of mass
        return self.flock_state[0].mean(0)

    @property
    def positions(self):
        return self.flock_state[0]

    @property
    def velocities(self):
        return self.flock_state[1]

    def _update_neighbour_masks(self):
        ali_r, coh_r, sep_r = self.boid_params[ForceTypes.RADIUS]
        self.sq_dist = np.square(self.positions[:,np.newaxis]-self.positions[np.newaxis,:]).sum(2)
        np.fill_diagonal(self.sq_dist, np.inf)
        self.align_neighbour_mask = self.sq_dist<=ali_r*ali_r
        self.cohesion_neighbour_mask = self.sq_dist<=coh_r*coh_r
        self.separation_neighbour_mask = self.sq_dist<=sep_r*sep_r

    def get_alignment(self):
        new_velocities = self.get_neighbours_average(BoidParams.ALIGNMENT, True)
        return self.convert_to_steer(new_velocities*self.boid_params[ForceTypes.WEIGHT][BoidParams.ALIGNMENT])

    def get_cohesion(self):
        new_velocities = self.get_neighbours_average(BoidParams.COHESION, True) - self.positions
        return self.convert_to_steer(new_velocities*self.boid_params[ForceTypes.WEIGHT][BoidParams.COHESION])

    def get_separation(self):
        mask = self.separation_neighbour_mask
        boid_count = len(self.flock_state[0])
        dim_count = self.flock_state.shape[-1]
        steer = ((self.positions[:,np.newaxis]-self.positions[np.newaxis,:])/self.sq_dist[...,np.newaxis])
        np.putmask(steer,np.repeat(~mask, 2).reshape(boid_count,boid_count,dim_count), 0)
        neighbour_count = mask.sum(1, keepdims=True)
        new_velocities = np.where((neighbour_count>0).flatten()[:,np.newaxis], steer.sum(1)/neighbour_count, np.zeros((boid_count,dim_count)))
        return self.convert_to_steer(new_velocities*self.boid_params[ForceTypes.WEIGHT][BoidParams.SEPARATION])

    def get_attraction(self):
        steer = self.current_waypoint - self.positions
        #steer /= np.linalg.norm(steer, axis=1, keepdims=True)
        return self.convert_to_steer(steer)*self.boid_params[ForceTypes.WEIGHT][BoidParams.ATTRACTION]

    def convert_to_steer(self,new_vel): # b. checking steer and speed
        new_v_mag = np.linalg.norm(new_vel, axis=1, keepdims=True)
        old_v_mag = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        new_v_unit = new_vel/new_v_mag
        old_v_unit = self.velocities/old_v_mag
        angle = np.empty(old_v_unit.shape[0])
        for i in range(len(old_v_unit)):
            angle[i] = np.arccos(np.dot(old_v_unit[i], new_v_unit[i]))
            
        old_v_unit_normals = np.array([-old_v_unit[:,1],old_v_unit[:,0]], dtype="float64")
        mag_limited = new_v_mag > MAX_SPEED
        steer_limited = angle > MAX_STEER
        for i in range(len(steer_limited)):
            if not steer_limited[i] and not mag_limited[i]:
                continue
            if steer_limited[i]:
                rot_mat = MAX_STEER_ROT_CW if np.dot(old_v_unit_normals.T[i], new_v_unit[i])<0 else MAX_STEER_ROT_CCW
                new_vel[i] = rot_mat @ old_v_unit[i]
            else:
                new_vel[i] = new_v_unit[i]
            new_vel[i] *= min(new_v_mag[i], MAX_SPEED)
        acc = new_vel - self.velocities
        for i, vec in enumerate(acc):
            force_mag_sq = np.dot(vec,vec)
            if force_mag_sq > MAX_FORCE_SQ:
                acc[i] = vec/np.sqrt(force_mag_sq)*MAX_FORCE
        return acc

    def get_neighbours_average(self, radius, velocity):
        mask = self.align_neighbour_mask if radius==BoidParams.ALIGNMENT else self.cohesion_neighbour_mask if radius==BoidParams.COHESION else self.separation_neighbour_mask
        boid_count = len(self.flock_state[0])
        dim_count = self.flock_state.shape[-1]
        neighbours = np.broadcast_to(self.flock_state[1 if velocity else 0, None, :],(boid_count,boid_count,dim_count)).copy()
        np.putmask(neighbours,np.repeat(~mask, 2).reshape(boid_count,boid_count,dim_count), 0)
        neighbour_count = mask.sum(1,keepdims=True)
        return np.where((neighbour_count>0).flatten()[:,np.newaxis], neighbours.sum(0)/neighbour_count, np.zeros((boid_count,dim_count)))

    def update_state(self, flock_state, boid_params=None):
        if self.flock_state.shape != flock_state.shape:
            raise ValueError("Flock state shape does not match")
        self.flock_state = flock_state
        if boid_params is not None:
            self.boid_params = boid_params
        self._update_neighbour_masks()
        diff = self.center_of_mass-self.current_waypoint
        if np.dot(diff, diff) < ENTITY_SIZE_SQ*10:#TODO magic number
            self.waypoint_idx = (self.waypoint_idx+1)%len(self.waypoints)
            self.reached_waypoint = np.zeros(len(self.reached_waypoint), dtype=bool)
        return self.waypoint_idx
