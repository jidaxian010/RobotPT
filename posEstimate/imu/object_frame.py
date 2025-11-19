import numpy as np

"""
Input: imu data (left, right, center)
Output: imu data in object frame
"""

IMU_to_OBJECT_R_right = np.array([[0, 0.939693, -0.34202],
                                    [-1,  0, 0],
                                    [0,  0.34202, 0.939693]]) # left_camera
IMU_to_OBJECT_T_right = np.array([-0.098085, -0.0175, 0.017882])

IMU_to_OBJECT_R_left = np.array([[0, 0.939693, 0.34202],
                                    [-1,  0, 0],
                                    [0,  -0.34202, 0.939693]]) # right_camera
IMU_to_OBJECT_T_left = np.array([0.098085, -0.0175, 0.017882])
IMU_to_OBJECT_R_vectornav = np.array([[0.7071, 0, -0.7071],
                                    [0.7071,  0, 0.7071],
                                    [0,  -1.0, 0]]) # vectornav
IMU_to_OBJECT_T_vectornav = np.array([0.069563, 0.061069, 0.201102])


class ObjectFrame:
    def __init__(self, topic_name):
        self.topic_name = topic_name
        
        # Switch to select correct transformation based on IMU topic
        if topic_name == "imu_left":
            self.imu_to_object_R = IMU_to_OBJECT_R_left
            self.imu_to_object_T = IMU_to_OBJECT_T_left
        elif topic_name == "imu_right":
            self.imu_to_object_R = IMU_to_OBJECT_R_right
            self.imu_to_object_T = IMU_to_OBJECT_T_right
        elif topic_name == "imu_vectornav":
            # Assuming vectornav is at the center, use identity transformation as placeholder
            # You can update these values with the actual transformation
            self.imu_to_object_R = IMU_to_OBJECT_R_vectornav
            self.imu_to_object_T = IMU_to_OBJECT_T_vectornav
        else:
            raise ValueError(f"Unknown IMU topic: {topic_name}. Supported topics: 'imu_left', 'imu_right', 'imu_vectornav'")
        

    def gyro_objectframe(self, gyro):
        """Transform gyroscope data from IMU frame to arm frame"""
        gyro_object = np.zeros_like(gyro)
        for i, gyro_imu in enumerate(gyro):
            gyro_object[i] = self.imu_to_object_R @ gyro_imu

        # window_size = 100
        # if len(gyro_object) > window_size:
        #     # Apply moving average filter
        #     for i in range(gyro_object.shape[1]):  # For each axis
        #         gyro_object[:, i] = np.convolve(gyro_object[:, i], np.ones(window_size)/window_size, mode='same')

        return gyro_object
    
    def accel_objectframe(self, acc, gyro):
        """
        Transform acceleration from IMU frame to arm frame.
        
        Complete transformation equation: a_arm = a_clean + angular_accel * T
        
      
        """
        R_matrix = self.imu_to_object_R
        T_vector = self.imu_to_object_T
        accel_object = acc @ R_matrix.T
        gyro_object = gyro @ R_matrix.T
        
        # Apply rotation to each row: (R @ a^T)^T = a @ R^T
        linear_accel = accel_object
        
        # Apply rotation to cross products
        angular_accel = np.gradient(gyro_object, axis=0)
        gyro_accel = np.cross(angular_accel, T_vector)
        centripetal_accel = np.cross(gyro_object, np.cross(gyro_object, T_vector))
        accel_object = linear_accel + gyro_accel + centripetal_accel

        return accel_object
    
    def process_data(self, acc, gyro):
        gyro_object = self.gyro_objectframe(gyro)
        accel_object = self.accel_objectframe(acc, gyro_object)
        return accel_object, gyro_object

    @classmethod
    def create_for_topic(cls, topic_name):
        """Factory method to create ObjectFrame for a specific IMU topic."""
        return cls(topic_name)

def main():
    acc = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    gyro = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    
    # Test different IMU topics
    for topic in ["imu_left", "imu_right", "imu_vectornav"]:
        print(f"\n=== Testing {topic} ===")
        try:
            object_frame = ObjectFrame(topic)
            accel_object, gyro_object = object_frame.process_data(acc, gyro)
            print(f"accel_object shape: {accel_object.shape}")
            print(f"gyro_object shape: {gyro_object.shape}")
        except Exception as e:
            print(f"Error with {topic}: {e}")

if __name__ == "__main__":
    main()

