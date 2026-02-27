import numpy as np
import matplotlib.pyplot as plt

def total_least_squares(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    U = np.vstack((x - x_mean, y - y_mean)).T
    
    UT_U = np.dot(U.T, U)

    eigenvalues, eigenvectors = np.linalg.eig(UT_U)
    
    min_idx = np.argmin(eigenvalues)
    a, b = eigenvectors[:, min_idx]
    
    d = a * x_mean + b * y_mean
    
    if d < 0:
        a, b, d = -a, -b, -d
        
    return a, b, d


print("--- Part (a): TLS on Line 1 ---")
try:
    D = np.genfromtxt("lines.csv", delimiter=",", skip_header=1)
    
    x1 = D[:, 0]
    y1 = D[:, 3]
    
    a1, b1, d1 = total_least_squares(x1, y1)
    print(f"Parameters for Line 1:")
    print(f"a = {a1:.4f}, b = {b1:.4f}, d = {d1:.4f}")
    print(f"Equation: {a1:.4f}x + {b1:.4f}y = {d1:.4f}\n")

except OSError:
    print("Error: 'lines.csv' not found. Please ensure it is in the same directory.")
    np.random.seed(42)
    D = np.random.rand(100, 6) * 10 - 5






print("--- Part (b): RANSAC for 3 lines ---")

X_cols = D[:, :3]
Y_cols = D[:, 3:]
X_all = X_cols.flatten()
Y_all = Y_cols.flatten()

points = np.vstack((X_all, Y_all)).T

def run_ransac(points, iterations=1000, distance_threshold=0.1):
    best_inliers_idx = []
    best_model = None
    
    for _ in range(iterations):
        idx = np.random.choice(len(points), 2, replace=False)
        sample = points[idx]
        
        a, b, d = total_least_squares(sample[:, 0], sample[:, 1])
        
        distances = np.abs(a * points[:, 0] + b * points[:, 1] - d)
        
        inliers = np.where(distances < distance_threshold)[0]
 
        if len(inliers) > len(best_inliers_idx):
            best_inliers_idx = inliers
            best_model = (a, b, d)
            
    final_a, final_b, final_d = total_least_squares(
        points[best_inliers_idx, 0], 
        points[best_inliers_idx, 1]
    )
    
    return (final_a, final_b, final_d), best_inliers_idx

found_lines = []
remaining_points = points.copy()

threshold = 0.5 

for i in range(3):
    if len(remaining_points) < 2:
        break 
        
    model, inliers_idx = run_ransac(remaining_points, iterations=500, distance_threshold=threshold)
    
    a, b, d = model
    found_lines.append(model)
    print(f"Found Line {i+1}: {a:.4f}x + {b:.4f}y = {d:.4f} (Inliers: {len(inliers_idx)})")
    
    remaining_points = np.delete(remaining_points, inliers_idx, axis=0)

print(f"Points left as outliers: {len(remaining_points)}")
