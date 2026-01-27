import math
import numpy as np
from src.envs.InfiltrationEnv import MAX_FORCE, MAX_FORCE_SQ, MAX_STEER, MAX_STEER_ROT_CCW, MAX_STEER_ROT_CW, MAX_STEER_SQ, BoidParams, ForceTypes, ENTITY_SIZE_SQ, MAX_SPEED, NEIGHBOURHOOD_MAX, NEIGHBOURHOOD_MIN

def dss_planning(env, agent):
    return None, None


# Adds points and fits curves with every imaginary boid and it's original's original model

class AI_machine_learner:

    poly_count = 5 # data points required for one polynomial
    maximum_concurrent_points = 40 # data points required for one polynomial

    def __init__(self, agent):
        self.prediction_errors = np.array([], dtype=np.float64)
        self.degree = 4
        #PolynomialCurveFitter fitter = PolynomialCurveFitter.create(4) # Fourth degree polynomial
        # lists of observed points for a parameter for a team
        self.obs = self.initialise_observed_point_lists()
        # lists of coefficients for fitted parameters for a team
        self.coeff_count = 0 #np.array([], dtype=np.float64)
        self.coeff = self.initialise_coeff_lists()
        # list of derivatives
        # self.derivative_sw, self.derivative_aw, self.derivative_cw, self.derivative_sns, self.derivative_ans, self.derivative_cns = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.derivatives = np.zeros((2,3), dtype=np.float64)
        self.parent_boid = agent # the adhoc agent holding this internal model
        self.internal_model_ref = agent.getInternal_model() # instance of Boid internal_model_ref
        self.error = 0 # meant to be int apparently
        self.points = np.zeros((2,2), dtype=np.float64) # the points to be fitted
        self.imaginary_pos, self.original_pos = (np.array([0.0, 0.0]), np.array([0.0, 0.0])) # the positions of the imaginary boid and the original boid
        
        
        self.cull_point_lists()

    def getPrediction_errors_for_team(self):
        return self.prediction_errors

    # public int setPrediction_error_for_team(int t, int in):
    #     return this.prediction_errors = in
    # }

    def calculate_error(self, b, points): # b was a simulated boid that wrapped the original boid, points is the positions of both
        if b is not None and b.getOriginal() is not None:
            imaginary_pos = points[0]
            original_pos = points[1]
            angle_diff1 = 10 * ((b.getOriginal().getVelocity().heading() - b.getVelocity().heading())) % 360
            angle_diff2 = 10 * ((b.getVelocity().heading() - b.getOriginal().getVelocity().heading())) % 360
            imaginary_pos -= original_pos
            return -(min(angle_diff1, angle_diff2)) + imaginary_pos.mag()
        return self.error

    def calculate_points(self, b):
        self.points[0][0] = b.getLocation().x
        self.points[0][1] = b.getLocation().y
        self.points[1][0] = b.getOriginal().getLocation().x
        self.points[1][1] = b.getOriginal().getLocation().y
        return self.points

    def initialise_observed_point_lists(self):
        self.obs = [[[],[],[]],[[],[],[]]]
        return self.obs

    def initialise_coeff_lists(self):
        self.coeff = np.zeros((2,3,5), dtype=np.float64)
        return self.coeff

    def update_estimates(self):
        self.derivatives.clip(NEIGHBOURHOOD_MIN, NEIGHBOURHOOD_MAX)
        # for derivative in derivatives:
        #     # some derivatives can be as large or small as infinity, this is not desired so
        #     # limit them
        #     derivatives[i] = Math.min(Math.max(derivatives[i], -AI_manager.getNeighbourhoodUpperBound()),
        #             AI_manager.getNeighbourhoodUpperBound())

        #self.parent_boid.getInternal_model().ai_s.learning_update(derivatives)

    def record_error_for_obs(self):
        params = self.internal_model_ref.boid_params
        self.obs[ForceTypes.RADIUS][BoidParams.ALIGNMENT].append(np.array(params[ForceTypes.RADIUS][BoidParams.ALIGNMENT],self.error))
        self.obs[ForceTypes.RADIUS][BoidParams.COHESION].append(np.array(params[ForceTypes.RADIUS][BoidParams.COHESION],self.error))
        self.obs[ForceTypes.RADIUS][BoidParams.SEPARATION].append(np.array(params[ForceTypes.RADIUS][BoidParams.SEPARATION],self.error))
        self.obs[ForceTypes.WEIGHT][BoidParams.ALIGNMENT].append(np.array(params[ForceTypes.WEIGHT][BoidParams.ALIGNMENT],self.error))
        self.obs[ForceTypes.WEIGHT][BoidParams.COHESION].append(np.array(params[ForceTypes.WEIGHT][BoidParams.COHESION],self.error))
        self.obs[ForceTypes.WEIGHT][BoidParams.SEPARATION].append(np.array(params[ForceTypes.WEIGHT][BoidParams.SEPARATION],self.error))
        self.coeff_count += 1
        # if (OutputWriter.isOutput_to_file()):
            # String data = param_sw + "," + error
            # OutputWriter.output_perspective(parent_boid.getTeam(), observed_t, data, "param_sw")
            # data = param_aw + "," + error
            # OutputWriter.output_perspective(parent_boid.getTeam(), observed_t, data, "param_aw")
            # data = param_cw + "," + error
            # OutputWriter.output_perspective(parent_boid.getTeam(), observed_t, data, "param_cw")
            # data = param_sns + "," + error
            # OutputWriter.output_perspective(parent_boid.getTeam(), observed_t, data, "param_sns")
            # data = param_ans + "," + error
            # OutputWriter.output_perspective(parent_boid.getTeam(), observed_t, data, "param_ans")
            # data = param_cns + "," + error
            # OutputWriter.output_perspective(parent_boid.getTeam(), observed_t, data, "param_cns")

    def record_polynomials_for_observation_of(self): #TODO this is wrong
        self.coeff[ForceTypes.RADIUS][BoidParams.ALIGNMENT] = np.polynomial.polynomial.polyfit(self.obs[ForceTypes.RADIUS][BoidParams.ALIGNMENT], self.obs[ForceTypes.RADIUS][BoidParams.ALIGNMENT], deg=self.degree)
        self.coeff[ForceTypes.RADIUS][BoidParams.COHESION] = np.polynomial.polynomial.polyfit(self.obs[ForceTypes.RADIUS][BoidParams.COHESION], self.obs[ForceTypes.RADIUS][BoidParams.COHESION], deg=self.degree)
        self.coeff[ForceTypes.RADIUS][BoidParams.SEPARATION] = np.polynomial.polynomial.polyfit(self.obs[ForceTypes.RADIUS][BoidParams.SEPARATION], self.obs[ForceTypes.RADIUS][BoidParams.SEPARATION], deg=self.degree)
        self.coeff[ForceTypes.WEIGHT][BoidParams.ALIGNMENT] = np.polynomial.polynomial.polyfit(self.obs[ForceTypes.WEIGHT][BoidParams.ALIGNMENT], self.obs[ForceTypes.RADIUS][BoidParams.ALIGNMENT], deg=self.degree)
        self.coeff[ForceTypes.WEIGHT][BoidParams.COHESION] = np.polynomial.polynomial.polyfit(self.obs[ForceTypes.WEIGHT][BoidParams.COHESION], self.obs[ForceTypes.WEIGHT][BoidParams.COHESION], deg=self.degree)
        self.coeff[ForceTypes.WEIGHT][BoidParams.SEPARATION] = np.polynomial.polynomial.polyfit(self.obs[ForceTypes.WEIGHT][BoidParams.SEPARATION], self.obs[ForceTypes.WEIGHT][BoidParams.SEPARATION], deg=self.degree)
        # if (OutputWriter.isOutput_to_file()):
            # String data = coeff_sw[4] + "," + coeff_sw[3] + "," + coeff_sw[2] + ","
                    # + coeff_sw[1] + "," + coeff_sw[0]
            # OutputWriter.output_perspective(parent_boid.getTeam(), observed_t, data, "poly_sw")
            # data = coeff_aw[4] + "," + coeff_aw[3] + "," + coeff_aw[2] + ","
                    # + coeff_aw[1] + "," + coeff_aw[0]
            # OutputWriter.output_perspective(parent_boid.getTeam(), observed_t, data, "poly_aw")
            # data = coeff_cw[4] + "," + coeff_cw[3] + "," + coeff_cw[2] + ","
                    # + coeff_cw[1] + "," + coeff_cw[0]
            # OutputWriter.output_perspective(parent_boid.getTeam(), observed_t, data, "poly_cw")
            # data = coeff_sns[4] + "," + coeff_sns[3] + "," + coeff_sns[2] + ","
                    # + coeff_sns[1] + "," + coeff_sns[0]
            # OutputWriter.output_perspective(parent_boid.getTeam(), observed_t, data, "poly_sns")
            # data = coeff_ans[4] + "," + coeff_ans[3] + "," + coeff_ans[2] + ","
                    # + coeff_ans[1] + "," + coeff_ans[0]
            # OutputWriter.output_perspective(parent_boid.getTeam(), observed_t, data, "poly_ans")
            # data = coeff_cns[4] + "," + coeff_cns[3] + "," + coeff_cns[2] + ","
                    # + coeff_cns[1] + "," + coeff_cns[0]
            # OutputWriter.output_perspective(parent_boid.getTeam(), observed_t, data, "poly_cns")

    # empties recorded data points to stop infinite accumilation
    def cull_point_lists(self):
        self.obs = np.zeros_like(self.obs, dtype=np.float64)
        self.coeff_count = 0

    # max_distance(): #TODO reimplement these constants
        # return (int) ((Constants.Boids.MAX_SPEED * 2) * Launcher.HISTORY_LENGTH)

    def create_new_term(self, exponent, coeffs, param_x):
        # term format... e.g. 5x^4 becomes 5(4x^5)
        # is is the point of the derivative
        # so becomes coeff(exponent*x^exponent)
        term = coeffs * (exponent * abs(param_x)**coeffs) # doe abs work?
        if (math.isinf(term)):
            term = 1
        return term if param_x > 0 else -term

    def calculate_derivative(self, param):# TODO reimplement
        param = self.internal_model_ref.boid_params
        np.zeros((2,3,5), dtype=np.float64)
    #     param_x = 0
    #     switch (param):
    #     case "sw":
    #         coeffs = coeff_sw
    #         param_x = parent_ai.getSeparationForceWeight()
    #         break
    #     case "aw":
    #         coeffs = coeff_aw
    #         param_x = parent_ai.getAlignmentForceWeight()
    #         break
    #     case "cw":
    #         coeffs = coeff_cw
    #         param_x = parent_ai.getCohesionForceWeight()
    #         break
    #     case "sns":
    #         coeffs = coeff_sns
    #         param_x = parent_ai.getSeparationForce()
    #         break
    #     case "ans":
    #         coeffs = coeff_ans
    #         param_x = parent_ai.getAlignForce()
    #         break
    #     case "cns":
    #         coeffs = coeff_cns
    #         param_x = parent_ai.getCohesionForce()
    #         break
    #     default:
    #         System.out.println("error: not a real term arguement")
    #         break
    #     }
    #     double terms_toal = 0
    #     for (int e = 1 e < 5 e++):
    #         # power of zero can be ignored for calculating the gradient
    #         terms_toal = terms_toal + create_new_term(e, coeffs[e], param_x)
    #     }
    #     return terms_toal
    # }

    def calculate_derivative(self): #TODO reimplement
        return
    #     derivative_sw = sum_terms("sw", observed_t)
    #     derivative_aw = sum_terms("aw", observed_t)
    #     derivative_cw = sum_terms("cw", observed_t)
    #     derivative_sns = sum_terms("ans", observed_t)
    #     derivative_ans = sum_terms("sns", observed_t)
    #     derivative_cns = sum_terms("cns", observed_t)
    # }

    # takes the final state of the imaginary flock
    def run(self, mind_flock):
        observer_t = self.agent.getTeam()
        for boid in mind_flock.getAllBoids():
            b = boid #need to figure out a way to replicate the cast
            observed_t = b.getOriginal().getTeam()
            points = self.calculate_points(b)
            error = self.calculate_error(b, points)
            # if error is > max travel speed it is a result of wrapping and can be ignored.
            # do not plot if error is equal to zero as no interations are implied
            # if (Launcher.isSim_drawtrails())
                #draw_error_bars(b, observer_t)
            if (observed_t != observer_t):
                if error < MAX_SPEED*2 and not b.isAlone(): # ((Constants.Boids.MAX_SPEED * 2) * Launcher.HISTORY_LENGTH)
                    self.record_error_for_obs(observed_t)
                if (self.coeff_count > self.poly_count):
                    self.record_polynomials_for_observation_of(observed_t) # do this periodically once data has accumilated
                    self.calculate_derivative(observed_t) # do this periodically once data has accumilated
                    self.update_estimates(observed_t) # acts on parent_boid
                if (self.coeff_count > self.maximum_concurrent_points):
                    self.cull_point_lists(observed_t) # empty data every n frames to stop memory leakage
