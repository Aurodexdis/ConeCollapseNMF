"""
.py file containing all functions needed to run CC_Pseudo_Inverse_Combinatorics.ipynb
"""

# Copyright (c) 2024 Aurod Ounsinegad.
#
# This is free, open software released under the MIT License.  See
# `LICENSE` or https://choosealicense.com/licenses/mit/ for details.

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import distance
import itertools

import plotly.graph_objects as go
from PIL import Image
import io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

###Define Functions for CC Algorithms###

def generate_data(m, n, r):
    """
    Generate random data on a single hyperplane within the non-negative 3D plane and plot interactively.

    Args:
    m: Number of dimensions
    n: Number of data points
    r: Rank of the generated data

    Returns:
    phi: Initial cone vectors (identity matrix)
    X: Generated data points
    """
    U = np.abs(np.random.randn(m, r))
    # Initialize U as the identity matrix (axes) [UNCOMMENT IF YOU WANT THE POINTS TO EFFECTIVELY BE SCATTERED ABOUT THE NONEGATIVE QUADRANT]
    # U = np.eye(m)
    V = np.abs(np.random.randn(r, n))
    X = U @ V

    # Initialize phi as the identity matrix
    phi = np.eye(m)

    mu = np.ones(m) / np.sqrt(m)

    fig = go.Figure()

    # Determine the maximum extent of the data
    max_extent = max(np.max(X), np.max(U)) * 1.2  # Extend 20% beyond the data

    # Plot the data points
    fig.add_trace(go.Scatter3d(
        x=X[0],
        y=X[1],
        z=X[2],
        mode='markers',
        marker=dict(
            size=4,
            color=X[2],  # color by z-coordinate
            colorscale='Viridis',
            opacity=0.8
        ),
        name='Data points'
    ))

    # Plot the vectors that define the hyperplane
    for i in range(phi.shape[1]):
        fig.add_trace(go.Scatter3d(
            x=[0, phi[0, i] * max_extent],
            y=[0, phi[1, i] * max_extent],
            z=[0, phi[2, i] * max_extent],
            mode='lines',
            line=dict(color='red', width=3),
            name=f'Vector {i+1}'
        ))

    # Set layout
    fig.update_layout(
        title='Generated Data on Hyperplane',
        scene=dict(
            xaxis=dict(range=[0, max_extent]),
            yaxis=dict(range=[0, max_extent]),
            zaxis=dict(range=[0, max_extent]),
            aspectmode='cube'
        ),
        width=800,
        height=600,
        autosize=False
    )

    # Show the plot
    fig.show()

    return phi, X

def pseudo_inverse_test(U, x):
    """Check if a point falls out of the cone using the pseudo-inverse test."""
    U_new = np.hstack((U, x.reshape(-1, 1)))
    v = np.linalg.pinv(U_new.T @ U_new) @ U_new.T @ x
    return np.any(v < 0)

def combinatorial_test(U, x):
    """Check if a point falls out of the cone using the combinatorial test."""
    m, n = U.shape
    for indices in itertools.combinations(range(n), m):
        U_subset = U[:, indices]
        try:
            v = np.linalg.solve(U_subset.T @ U_subset, U_subset.T @ x)
            if np.all(v >= 0):
                return False  # Point is inside the cone
        except np.linalg.LinAlgError:
            continue  # Subset is not linearly independent, skip
    return True  # Point is outside the cone

def plot_cone_and_points(U, X, title):
    """Plot the cone (created by vectors) and all points within it interactively."""
    fig = go.Figure()

    # Determine the maximum extent of the data
    max_extent = max(np.max(X), np.max(U)) * 1.2  # Extend 20% beyond the data

    # Plot the cone vectors
    for i in range(U.shape[1]):
        fig.add_trace(go.Scatter3d(
            x=[0, U[0, i] * max_extent],
            y=[0, U[1, i] * max_extent],
            z=[0, U[2, i] * max_extent],
            mode='lines',
            line=dict(color='red', width=3),
            name=f'Vector {i+1}'
        ))

    # Plot the points
    fig.add_trace(go.Scatter3d(
        x=X[0, :],
        y=X[1, :],
        z=X[2, :],
        mode='markers',
        marker=dict(size=4, color='blue'),
        name='Points'
    ))

    # Set layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(range=[0, max_extent]),
            yaxis=dict(range=[0, max_extent]),
            zaxis=dict(range=[0, max_extent]),
            aspectmode='cube'
        ),
        width=800,
        height=600,
        autosize=False
    )

    # Show the plot
    fig.show()

def plot_cone_points_and_bounds(U, X, title):
    """
    Plot the cone (created by vectors and planes), all points within it, and the mu vector interactively.

    Args:
    U: Cone vectors
    X: Data points
    title: Title of the plot
    """
    fig = go.Figure()

    # Determine the maximum extent of the data
    max_extent = max(np.max(X), np.max(U)) * 1.2  # Extend 20% beyond the data

    # Calculate mu (center of the nonnegative orthant)
    m = X.shape[0]  # number of dimensions
    mu = np.ones(m) / np.sqrt(m)
    mu_extended = mu * max_extent  # Extend mu to the edge of the plot

    # Plot the mu vector
    fig.add_trace(go.Scatter3d(
        x=[0, mu_extended[0]],
        y=[0, mu_extended[1]],
        z=[0, mu_extended[2]],
        mode='lines',
        line=dict(color='green', width=3),
        name='μ (center of nonnegative orthant)'
    ))

    # Plot the cone vectors
    for i in range(U.shape[1]):
        fig.add_trace(go.Scatter3d(
            x=[0, U[0, i] * max_extent],
            y=[0, U[1, i] * max_extent],
            z=[0, U[2, i] * max_extent],
            mode='lines',
            line=dict(color='red', width=3),
            name=f'Vector {i+1}'
        ))

    # Plot the planes of the cone
    for i in range(U.shape[1]):
        for j in range(i+1, U.shape[1]):
            plane_points = np.column_stack(([0, 0, 0], U[:, i] * max_extent, U[:, j] * max_extent))
            fig.add_trace(go.Mesh3d(
                x=plane_points[0],
                y=plane_points[1],
                z=plane_points[2],
                opacity=0.2,
                color='red',
                hoverinfo='none'
            ))

    # Plot the points
    fig.add_trace(go.Scatter3d(
        x=X[0, :],
        y=X[1, :],
        z=X[2, :],
        mode='markers',
        marker=dict(size=4, color='blue'),
        name='Points'
    ))

    # Set layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(range=[0, max_extent]),
            yaxis=dict(range=[0, max_extent]),
            zaxis=dict(range=[0, max_extent]),
            aspectmode='cube'
        ),
        width=800,
        height=600,
        autosize=False
    )

    # Show the plot
    fig.show()

def update_vector(v, X, mu, theta):
    """
    Update a vector by rotating it towards mu (center of the nonnegative orthant) and considering outside points.

    Args:
    v: Vector to update
    X: Data points
    mu: Center of the nonnegative orthant
    theta: Step size for the update
    """
    # Find points outside the cone
    outside_points = [X[:, i] for i in range(X.shape[1]) if pseudo_inverse_test(v.reshape(-1, 1), X[:, i]) and combinatorial_test(v.reshape(-1, 1), X[:, i])]

    if outside_points:
        # Calculate the mean direction of outside points
        mean_direction = np.mean(outside_points, axis=0)
        mean_direction /= np.linalg.norm(mean_direction)

        # Combine with the direction towards mu
        update_direction = 0.7 * mean_direction + 0.3 * mu
        update_direction /= np.linalg.norm(update_direction)
    else:
        update_direction = mu - v
        update_direction /= np.linalg.norm(update_direction)

    # Update the vector
    updated_v = v + np.sin(theta) * update_direction
    return updated_v / np.linalg.norm(updated_v)

def rotate_vector_around_axis(vector, axis, angle):
    """
    Rotate a vector around an arbitrary axis.

    Args:
    vector: Vector to rotate
    axis: Axis of rotation
    angle: Angle of rotation in radians
    """
    # Normalize the axis
    axis = axis / np.linalg.norm(axis)

    # Rodriguez rotation formula
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)

    return (vector * cos_theta +
            np.cross(axis, vector) * sin_theta +
            axis * np.dot(axis, vector) * (1 - cos_theta))

def cc_add(U, X, k, theta, max_iterations=1000):
    """
    Cone Collapsing algorithm with alternating vector updates and new vector addition.

    Args:
    U: Initial cone vectors
    X: Data points
    k: Number of new vectors to add (maximum)
    theta: Step size for vector updates
    max_iterations: Maximum number of iterations before breaking
    """
    m, n = X.shape
    iteration = 0
    update_index = 0
    new_vectors_added = 0

    # Define mu as the center of the nonnegative orthant
    mu = np.ones(m) / np.sqrt(m)

    while new_vectors_added < k and iteration < max_iterations:
        iteration += 1

        # Update a vector (rotating through the first 3 vectors)
        update_index = (update_index + 1) % 3
        U[:, update_index] = update_vector(U[:, update_index], X, mu, theta)

        # Check all points
        for i in range(n):
            x = X[:, i]
            if pseudo_inverse_test(U, x) and combinatorial_test(U, x):
                U = np.column_stack((U, x))
                new_vectors_added += 1
                # plot_cone_and_points(U, X, f"Iteration {iteration}: Added vector {new_vectors_added}")
                break  # Only add one vector per iteration

        if new_vectors_added == k:
            break

    return U, iteration

def cc_rotate(U, X, theta, initial_rotation_angle, max_iterations=1000, max_rotations=1000, convergence_threshold=1e-6):
    """
    Cone Collapsing algorithm with rotating cone approach and improved convergence criteria.

    Args:
    U: Initial cone vectors
    X: Data points
    theta: Step size for vector updates
    initial_rotation_angle: Initial angle to rotate the cone when a point falls out
    max_iterations: Maximum number of iterations before breaking
    max_rotations: Maximum number of rotations before reducing the rotation angle
    convergence_threshold: Threshold for convergence based on change in cone position
    """
    m, n = X.shape
    iteration = 0
    update_index = 0
    rotation_angle = initial_rotation_angle

    # Define mu as the center of the nonnegative orthant
    mu = np.ones(m) / np.sqrt(m)

    # For tracking convergence
    previous_U = U.copy()

    # For animation
    animation_frames = []
    overall_iteration = 0

    while iteration < max_iterations:
        iteration += 1
        update_index = (update_index + 1) % 3

        # Store previous U vectors
        previous_U = U.copy()

        # Try updating the vector
        U[:, update_index] = update_vector(U[:, update_index], X, mu, theta)

        # Check if any point falls out
        point_outside = any(pseudo_inverse_test(U, X[:, i]) and combinatorial_test(U, X[:, i]) for i in range(n))

        # Count points outside the cone
        points_outside = sum(pseudo_inverse_test(U, X[:, i]) and combinatorial_test(U, X[:, i]) for i in range(n))

        # Store frame for animation
        animation_frames.append((U.copy(), 0, rotation_angle, points_outside, overall_iteration))

        if point_outside:
            # Try rotating and updating
            for rotation_iter in range(max_rotations):
                # Create a rotation object
                rotation_vector = rotation_angle * mu
                rotation = R.from_rotvec(rotation_vector)

                # Rotate the entire cone
                U = rotation.apply(U.T).T

                # Check if any point falls out after rotation and update
                point_outside = any(pseudo_inverse_test(U, X[:, i]) and combinatorial_test(U, X[:, i]) for i in range(n))

                # Count points outside the cone
                points_outside = sum(pseudo_inverse_test(U, X[:, i]) and combinatorial_test(U, X[:, i]) for i in range(n))

                # Store frame for animation
                animation_frames.append((U.copy(), rotation_iter, rotation_angle, points_outside, overall_iteration))

                if not point_outside:
                    # Successful update after rotation
                    break

            if point_outside:
                # If still unsuccessful after max_rotations, reduce rotation angle
                rotation_angle /= 2
                U = previous_U  # Revert to the state before rotation attempts

        # Ensure all vectors remain nonnegative
        U = np.maximum(U, 0)

        # Check for convergence
        change = np.linalg.norm(U - previous_U)
        if change < convergence_threshold:
            print(f"Converged after {iteration} iterations.")
            break

        previous_U = U.copy()
        overall_iteration += 1

    return U, iteration, animation_frames

def cc_rotation_animation(X, animation_frames, output_filename='enhanced_cone_rotation.gif', frame_skip=1, frame_duration=200):
    """
    Create an enhanced animation of the cone rotation process with extended vectors, planes, and additional information.

    Args:
    X: Data points
    animation_frames: List of (U, rotation_iter, rotation_angle, points_outside, overall_iteration) tuples
    output_filename: Name of the output GIF file
    frame_skip: Number of frames to skip (to reduce file size and processing time)
    frame_duration: Duration of each frame in milliseconds
    """
    images = []
    max_val = np.max(X) * 1.2

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Calculate mu (center of the nonnegative orthant)
    m = X.shape[0]  # number of dimensions
    mu = np.ones(m) / np.sqrt(m)
    mu_extended = mu * max_val  # Extend mu to the edge of the plot

    for i, (U, rotation_iter, rotation_angle, points_outside, overall_iteration) in enumerate(animation_frames):
        if i % frame_skip != 0:
            continue

        ax.clear()

        # Plot data points
        ax.scatter(X[0], X[1], X[2], c='blue', s=20, alpha=0.6)

        # Plot extended cone vectors
        for j in range(U.shape[1]):
            vector = U[:, j]
            magnitude = np.linalg.norm(vector)
            if magnitude > 0:
                extended_vector = vector * (max_val / magnitude)
                ax.plot([0, extended_vector[0]], [0, extended_vector[1]], [0, extended_vector[2]], 'r-', linewidth=2)

        # Plot the planes of the cone
        for j in range(U.shape[1]):
            for k in range(j+1, U.shape[1]):
                plane_points = np.column_stack(([0, 0, 0], U[:, j] * max_val, U[:, k] * max_val))
                poly = Poly3DCollection([plane_points.T], alpha=0.2, facecolor='red')
                ax.add_collection3d(poly)

        # Plot mu as a green line
        ax.plot([0, mu_extended[0]], [0, mu_extended[1]], [0, mu_extended[2]], 'g-', linewidth=2, label='μ')

        # Set axis limits
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.set_zlim(0, max_val)

        # Add labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Add title with additional information
        title = f'Iteration: {overall_iteration}\n'
        title += f'Rotation Iteration: {rotation_iter}\n'
        title += f'Rotation Angle: {rotation_angle:.4f}\n'
        title += f'Points Outside Cone: {points_outside}'
        ax.set_title(title)

        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        images.append(Image.open(buf).copy())
        buf.close()

    plt.close(fig)

    # Save as GIF
    if images:
        images[0].save(output_filename, save_all=True, append_images=images[1:], duration=frame_duration, loop=0)
        print(f"Animation saved as {output_filename}")
    else:
        print("No frames were generated. Try reducing the frame_skip value.")

    # Close all images
    for img in images:
        img.close()

###Trying a Distance-Based Vector Update Approach###

def choose_vector(U, X):
    """
    Find the vector with the greatest distance to its closest point.

    Args:
    U: Current cone vectors
    X: Data points

    Returns:
    index: Index of the vector to update
    min_distance: Distance to the closest point
    closest_point: The closest point to the chosen vector
    """
    max_min_distance = -np.inf
    index_to_update = -1
    closest_point = None

    for i in range(U.shape[1]):
        distances = distance.cdist([U[:, i]], X.T, metric='euclidean')[0]
        min_distance = np.min(distances)
        if min_distance > max_min_distance:
            max_min_distance = min_distance
            index_to_update = i
            closest_point = X[:, np.argmin(distances)]

    return index_to_update, max_min_distance, closest_point

def update_vector_distance(U, X, mu, max_step_size):
    """
    Update the vector with the greatest distance to its closest point.

    Args:
    U: Current cone vectors
    X: Data points
    mu: Center of the nonnegative orthant
    max_step_size: Maximum allowed step size

    Returns:
    U_updated: Updated cone vectors
    """
    index_to_update, distance, closest_point = choose_vector(U, X)
    v = U[:, index_to_update]

    # Calculate direction towards the closest point
    direction = closest_point - v
    direction /= np.linalg.norm(direction)

    # Calculate adaptive step size
    step_size = min(distance, max_step_size)

    # Update the vector
    v_updated = v + step_size * direction

    # Ensure the updated vector remains in the nonnegative orthant
    v_updated = np.maximum(v_updated, 0)
    v_updated /= np.linalg.norm(v_updated)

    U_updated = U.copy()
    U_updated[:, index_to_update] = v_updated

    return U_updated

def cc_rotate_dist(U, X, initial_max_step_size, initial_rotation_angle, max_iterations=1000, max_rotations=1000, convergence_threshold=1e-6):
    """
    Cone Collapsing algorithm with distance-based vector updates and adaptive step size.

    Args:
    U: Initial cone vectors
    X: Data points
    initial_max_step_size: Initial maximum step size for vector updates
    initial_rotation_angle: Initial angle to rotate the cone when a point falls out
    max_iterations: Maximum number of iterations before breaking
    max_rotations: Maximum number of rotations before reducing the rotation angle
    convergence_threshold: Threshold for convergence based on change in cone position

    Returns:
    U: Final cone vectors
    iteration: Number of iterations performed
    animation_frames: List of frames for animation
    """
    m, n = X.shape
    iteration = 0
    rotation_angle = initial_rotation_angle
    max_step_size = initial_max_step_size

    # Define mu as the center of the nonnegative orthant
    mu = np.ones(m) / np.sqrt(m)

    previous_U = U.copy()
    animation_frames = []
    overall_iteration = 0

    while iteration < max_iterations:
        iteration += 1

        # Store previous U vectors for convergence check
        previous_U = U.copy()

        # Update the vector with the greatest distance to its closest point
        U = update_vector_distance(U, X, mu, max_step_size)

        # Check if any point falls out of the cone
        point_outside = any(pseudo_inverse_test(U, X[:, i]) and combinatorial_test(U, X[:, i]) for i in range(n))

        # Count points outside the cone
        points_outside = sum(pseudo_inverse_test(U, X[:, i]) and combinatorial_test(U, X[:, i]) for i in range(n))

        # Store frame for animation
        animation_frames.append((U.copy(), 0, rotation_angle, points_outside, overall_iteration))

        if point_outside:
            # Try rotating and updating
            for rotation_iter in range(max_rotations):
                # Create a rotation object
                rotation_vector = rotation_angle * mu
                rotation = R.from_rotvec(rotation_vector)

                # Rotate the entire cone
                U = rotation.apply(U.T).T

                # Check if any point falls out after rotation
                point_outside = any(pseudo_inverse_test(U, X[:, i]) and combinatorial_test(U, X[:, i]) for i in range(n))

                # Count points outside the cone
                points_outside = sum(pseudo_inverse_test(U, X[:, i]) and combinatorial_test(U, X[:, i]) for i in range(n))

                # Store frame for animation
                animation_frames.append((U.copy(), rotation_iter, rotation_angle, points_outside, overall_iteration))

                if not point_outside:
                    # Successful update after rotation
                    break

            if point_outside:
                # If still unsuccessful after max_rotations, reduce rotation angle and step size
                rotation_angle /= 2
                max_step_size /= 2
                U = previous_U  # Revert to the state before rotation attempts

        # Ensure all vectors remain nonnegative
        U = np.maximum(U, 0)

        # Check for convergence
        change = np.linalg.norm(U - previous_U)
        if change < convergence_threshold:
            print(f"Converged after {iteration} iterations.")
            break

        previous_U = U.copy()
        overall_iteration += 1

    return U, iteration, animation_frames
