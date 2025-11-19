import numpy as np

class FindPosition:
    def __init__(self, acc_object, orientation_R, dt, v0 = None, p0 = None, g_world = None, bias_correction_samples = 50, enable_bias_correction = True):
        """
        acc_object: (N,3) accelerometer specific force in body frame (without gravity removed)
        orientation_R: (N,3,3) rotation matrices R_WB mapping body->world
        dt: scalar timestep OR (N-1,) array
        v0: initial velocity in world frame (default zeros)
        p0: initial position in world frame (default zeros)
        g_world: gravity vector in world frame (default [0,0,-9.81])
        bias_correction_samples: number of initial samples to use for bias estimation (default 50)
        """
        self.acc_object = np.asarray(acc_object)
        self.orientation_R = np.asarray(orientation_R)
        self.N = self.acc_object.shape[0]
        self.dt = np.asarray(dt) if np.ndim(dt) else np.full(self.N-1, float(dt))

        self.v0 = np.zeros(3) if v0 is None else np.asarray(v0)
        self.p0 = np.zeros(3) if p0 is None else np.asarray(p0)
        self.g_world = np.array([0,0,-9.81]) if g_world is None else np.asarray(g_world)
        self.bias_correction_samples = bias_correction_samples
        self.enable_bias_correction = enable_bias_correction
        
    def compute_bias_correction(self, acc_specific_world):
        """
        Compute systematic bias from initial samples when sensor should be stationary.
        Returns bias vector to subtract from all acceleration measurements.
        """
        # Use first N samples for bias estimation
        n_samples = min(self.bias_correction_samples, len(acc_specific_world))
        
        # Compute expected gravity in world frame for each sample
        expected_gravity = np.tile(self.g_world, (n_samples, 1))  # Shape: (n_samples, 3)
        
        # Compute bias as difference between measured and expected gravity
        bias = np.mean(acc_specific_world[:n_samples] - expected_gravity, axis=0)
        
        print(f"Bias correction computed from {n_samples} samples:")
        print(f"  Measured gravity (avg): {np.mean(acc_specific_world[:n_samples], axis=0)}")
        print(f"  Expected gravity: {self.g_world}")
        print(f"  Computed bias: {bias}")
        print(f"  Bias magnitude: {np.linalg.norm(bias):.6f} m/s²")
        
        return bias
        
    def find_position(self):
        # Rotate accelerometer readings (specific force) into world frame
        acc_specific_world = (self.orientation_R @ self.acc_object[..., None]).squeeze(-1)
        
        # Compute and apply bias correction
        if self.enable_bias_correction:
            bias = self.compute_bias_correction(acc_specific_world)
            acc_specific_world_corrected = acc_specific_world - bias
        else:
            acc_specific_world_corrected = acc_specific_world
        
        # Convert specific force to true acceleration by subtracting gravity
        # acc_object measures upward force against gravity, so we subtract gravity to get true motion
        acc_world = acc_specific_world_corrected - self.g_world
        
        v_world = np.zeros((self.N,3))
        p_world = np.zeros((self.N,3))
        v_world[0] = self.v0
        p_world[0] = self.p0

        for k in range(self.N-1):
            dt_k = self.dt[k]
            # Velocity update (trapezoidal)
            v_world[k+1] = v_world[k] + 0.5*(acc_world[k] + acc_world[k+1]) * dt_k
            # Position update (trapezoidal)
            p_world[k+1] = p_world[k] + 0.5*(v_world[k] + v_world[k+1]) * dt_k

        return p_world, v_world, acc_world
