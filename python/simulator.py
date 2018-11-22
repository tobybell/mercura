#!/usr/local/bin/python3
import numpy as np

num_bodies = 2
duration = 365 * 24 * 60 * 60
timestep = 1
sgp = np.array([1.327124400189e20, 3.9860044188e14])
dynamics = np.zeros((num_bodies, 3, 3)) # position, velocity, acceleration
position = np.array([[-4.4910405450108775e5, 0, 0], [1.4952741801817593e11, 0, 0]])
velocity = np.array([[-4.4569063372411534e-10, -8.947881775228357e-2, 0], [1.4839093307604534e-4, 2.9791618338162836e4, 0]])
# Update Acceleration
#   Sum of gravitational force between all bodies
# Update Velocity
#   Velocity + timestep * acceleration
# Update Position
#   Position + timestep * velocity
# distances[i,j,:] is the vector from j to i
t = 0
while t < duration:
    distances = np.reshape(position, (num_bodies, 1, 3)) - np.reshape(position, (1, num_bodies, 3)) 
    magnitudes = np.sum(distances * distances, axis=2) + np.eye(num_bodies)
    accelerations = (sgp[:,None] / (magnitudes ** (3/2)) * (1 - np.eye(num_bodies)))[:,:,None] * distances
    accelerations = np.sum(accelerations, axis=0)

    velocity += accelerations * timestep
    position += velocity * timestep
    # print(np.linalg.norm(position[1]), np.arctan2(position[1,1], position[1,0]))
    if (t + timestep) % (24 * 60 * 60) < t % (24 * 60 * 60):
        print(t)
    t += timestep
print(position)
print(velocity)
# acc_i += sgp_j*normalized_distance_vector
#dynamics[:,1,:] = timestep * dynamics[:,2,:]
#dynamics[:,0,:] = timestep * dynamics[:,1,:]