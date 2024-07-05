# read the data from the npy file
import numpy as np
points_trajectories = np.load("points_trajectories_rigid.npy")
# points_trajectories = np.load("points_trajectories_spring_fake_rigid.npy")
# points_trajectories = np.load("points_trajectories_spring.npy")
points_trajectories = points_trajectories - points_trajectories[0] 
n = points_trajectories.shape[0]
n = 50
# randomly select n trajectories
idx = np.random.choice(points_trajectories.shape[0], n, replace=False)

points_lengths = np.linalg.norm(points_trajectories, axis=2)
points_lengths -= points_lengths[:, 0][:, np.newaxis]

# Draw the frequncy figure of point_lengths
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
for i in range(n):
    plt.plot(range(points_lengths.shape[1]), points_lengths[i])

# Add labels and title
plt.xlabel('Frame Index')
plt.ylabel('Distance')
plt.ylim(-0.05, 0.05)
# plt.title('Lines Representing Heights of Points')
plt.legend()
plt.show()