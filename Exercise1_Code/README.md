# TP Computer Graphics - Animation - M2 AI
## Author: Nguyen Nam Khanh - Nguyen Ho Minh Khanh

### 1.1 Forward kinematics
## a)

#### `compute_local_coordinates`

According to the course slide, we have the relationship: `Mi_global = Mpi_global * Mi_local` where `Mi_global` is the matrix of global coordinates of frame `i`, `Mpi_global` is the matrix of global coordinates of parent of `i` and `Mi_local` is the matrix of local coordinates of `i`.

So to compute the local coordinates of frame `i`, we used `np.linalg.solve(Mpi_global, Mi_global)` which computes `inverse(Mpi_global) * Mi_global`.

If frame `i` is the root in the tree, we return its global coordinates.

## b)

#### `update_skeleton_using_forward_kinematics`

This function rotates the selected frame so it points toward the target clicked in the scene. The target is given in world coordinates, while the skeleton is stored as local coordinates. The function computes the global coordinates then converts back to the selected frame's local coordinates.

The steps are:

- Copy the input pose: `updated_local_coordinates` starts as a copy of `local_coordinates`. The function changes only the selected frame's local coordinates; all other frames' local coordinates remain untouched.
- Compute the current world coordinates: `compute_global_coordinates(local_coordinates, parents)` gives the selected frame's current world origin and its parent's global coordinates.
- Find the direction to the target:
Our transform matrix looks like this:
[ r11  r12  tx ]
[ r21  r22  ty ]
[  0    0    1 ]
The top-left 2 by 2 matrix defines the rotation. The selected frame's origin is the translation in the last column of its 3×3 homogeneous transform: `global_coordinates[i, 0:2, 2]`. We get the direction by subtracting this from the origin.
- Handle target near the origin: If the direction has length at most `1e-12`, which means the user clicks too close to the origin, its orientation is undefined, so the function skips the rotation and returns the copied pose unchanged.
- Compute the desired angle: `np.arctan2(direction[1], direction[0])` gives the angle between x axis and the target direction. Its cos and sin build the 2D rotation matrix.
- Build the desired world coordinates: The rotation matrix is placed in the top-left 2×2 part of a 3×3 homogeneous transform. 
- Convert the desired world coordinates to local coordinates: For a root frame, the local and world coordinates are the same, so the desired coordinates is assigned directly. For a child frame, the hierarchy relation is `G_i = G_parent @ L_i`; solving this relation with `np.linalg.solve(G_parent, G_i)` gives the selected frame's new local coordinates without explicitly computing a matrix inverse.
- Return the updated local pose: The caller can recompute all global coordinates from this pose. Descendants keep their local coordinates and therefore follow the rotated frame through the parent hierarchy.

Here, `G_i` denotes frame `i`'s global (world) coordinates, `G_parent` the parent's global coordinates, and `L_i` frame `i`'s local coordinates. 

### 1.2 Inverse kinematics

#### `update_skeleton_using_FABRIC`

```python
update_skeleton_using_FABRIC(local_coordinates, parents, target_position)
```

Moves the end of the third-finger chain (frame `5`) toward a target using the
FABRIK inverse-kinematics method. The function builds the chain from frame `5`
back to the root, computes its joint positions and segment lengths, then
iteratively adjusts the joint positions with a forward pass and a backward
pass. The root stays fixed and the segment lengths are preserved.

If the target is farther from the root than the chain's total length, the
segments are aligned toward the target; in that case the end cannot reach it.
If the chain has effectively zero total length, the input pose is returned
unchanged. The target's first two coordinates (`x`, `y`) are used.

After solving the joint positions, the function converts the resulting chain
directions into local rotations and returns an updated copy of
`local_coordinates`. It does not modify the input array in place.

### 1.3 Linear blend skinning

#### `update_skin_vertices`

```python
update_skin_vertices(
    skin_vertex_positions,
    local_coordinates_init_pose,
    local_coordinates_current_pose,
    parents,
)
```

Deforms the skin vertices from the initial pose to the current skeleton pose
using linear blend skinning. It first computes each frame's global transform in
both poses. For frame `f`, it forms the transform from the initial pose to the
current pose:

```text
skin_transforms_f = T_f_current @ inverse(T_f_initial)
```

For each vertex, the function finds the nearest bone segment and the nearest
skeleton frame. It assigns inverse-distance weights to the parent frame of the
nearest bone and to the nearest frame, normalizing the two weights to sum to
one. If a vertex is exactly at a frame, that frame receives all the weight.
Each frame transform is applied to the vertex in homogeneous coordinates, and
the results are blended using those weights.