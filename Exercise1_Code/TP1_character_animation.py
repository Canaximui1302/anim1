import numpy as np
import polyscope as ps
import polyscope.imgui as psim

# Compute the generalized global coordinates for generalized local coordinates and parent tree
def compute_global_coordinates(local_coordinates, parents):
  nframes = local_coordinates.shape[0]
  global_coordinates = np.zeros((nframes,3,3), np.float64)
  global_coordinates[0] = local_coordinates[0]
  for i in range(1, nframes):
      global_coordinates[i] = global_coordinates[parents[i]] @ local_coordinates[i]
      
  return global_coordinates

# Compute the generalized local coordinates for generalized global coordinates and parent tree
def compute_local_coordinates(global_coordinates, parents):
  nframes = global_coordinates.shape[0]
  local_coordinates = np.zeros((nframes,3,3), np.float64)
  local_coordinates[0] = global_coordinates[0]
  for i in range(1, nframes):
     local_coordinates[i] = np.linalg.solve(
        global_coordinates[parents[i]], 
        global_coordinates[i]
    )
  return local_coordinates

# Given a target position, rotate the selected bone to that it is aligned with the line [bone_start_position target_point]
def update_skeleton_using_forward_kinematics(local_coordinates, parents, selected_bone_index, target_position):
  print("updating skeleton using forward kinematics...")
  updated_local_coordinates = np.copy(local_coordinates)
  
  global_coordinates = compute_global_coordinates(local_coordinates, parents)
  i = selected_bone_index

  # Frame origin in world coordinates; target_position may include a z component.
  start = global_coordinates[i, 0:2, 2] # the pivot
  direction = np.asarray(target_position)[0:2] - start # target direction

  if np.linalg.norm(direction) > 1e-12:
      angle = np.arctan2(direction[1], direction[0])
      c = np.cos(angle)
      s = np.sin(angle)

      desired_global = np.array([
          [c, -s, start[0]],
          [s,  c, start[1]],
          [0,  0, 1],
      ])

      parent = parents[i]
      if parent == -1:
          updated_local_coordinates[i] = desired_global
      else:
          # G_i = G_parent @ L_i, so L_i = solve(G_parent, G_i)
          updated_local_coordinates[i] = np.linalg.solve(
              global_coordinates[parent],
              desired_global,
          )

  return updated_local_coordinates
  
# Given a target position for the third finger, update the skeleton using the FABRIC algorithm so that the finger touches the target
def update_skeleton_using_FABRIC(local_coordinates, parents, target_position):
  print("updating skeleton using FABRIC...")
  updated_local_coordinates = np.copy(local_coordinates)

  # The third finger is frame 5; its ancestors form the arm-to-finger chain.
  end_index = 5
  chain = [end_index]
  while parents[chain[-1]] != -1:
    chain.append(int(parents[chain[-1]]))
  chain.reverse()

  global_coordinates = compute_global_coordinates(local_coordinates, parents)
  # global_coordinates =
  #   [R_00 R_01 t_x
  #    R_10 R_11 t_y
  #    0    0    1  ]
  points = global_coordinates[chain, 0:2, 2].copy()

  # target_position = (x, y, z)
  target = np.asarray(target_position, dtype=np.float64).reshape(-1)[:2]
  segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
  total_length = np.sum(segment_lengths)
  root = points[0].copy()

  if total_length <= 1e-12:
    return updated_local_coordinates

  target_distance = np.linalg.norm(target - root)
  
  # align the corresponding frames into a straight line from root to target
  if target_distance >= total_length:
    direction = target - root
    if target_distance > 1e-12:
      direction /= target_distance
    else:
      direction = np.array([1.0, 0.0])
    for i, length in enumerate(segment_lengths):
      points[i + 1] = points[i] + direction * length
  
  else:
    tolerance = 1e-6
    # iteration
    for _ in range(100):
      points[-1] = target
      # forward direction
      for i in range(len(points) - 2, -1, -1):
        direction = points[i] - points[i + 1]
        distance = np.linalg.norm(direction)
        if distance > 1e-12:
          points[i] = points[i + 1] + (direction / distance) * segment_lengths[i]

      points[0] = root
      # backward direction
      for i in range(len(points) - 1):
        direction = points[i + 1] - points[i]
        distance = np.linalg.norm(direction)
        if distance > 1e-12:
          points[i + 1] = points[i] + (direction / distance) * segment_lengths[i]

      if np.linalg.norm(points[-1] - target) <= tolerance:
        break

  # Record local coordinates
  desired_global_angles = {}
  for i in range(len(chain) - 1):
    parent_index = chain[i]
    child_index = chain[i + 1]
    
    direction = points[i + 1] - points[i]
    # direction = (x,y)
    
    child_offset = local_coordinates[child_index, 0:2, 2]
    # child_offset = (x,y)
    
    desired_global_angles[parent_index] = (
        np.arctan2(direction[1], direction[0])
        - np.arctan2(child_offset[1], child_offset[0])
    )
    # desired_global_angles = The global angle by which the parent frame must be 
    # rotated to maintain the child frame's local angle, while remaining consistent 
    # with the skeletal structure resolved above.

  for _, frame_index in enumerate(chain[:-1]):
    parent_index = parents[frame_index]
    parent_angle = (
        0.0 if parent_index == -1
        else desired_global_angles[parent_index]
    )
    local_angle = desired_global_angles[frame_index] - parent_angle
    c = np.cos(local_angle)
    s = np.sin(local_angle)
    updated_local_coordinates[frame_index, 0:2, 0:2] = np.array([
        [c, -s],
        [s,  c],
    ])

  return updated_local_coordinates
 
# Update the positions of the skin vertices using linear blend skinning
def update_skin_vertices(skin_vertex_positions, local_coordinates_init_pose, local_coordinates_current_pose, parents):
  print("updating skin vertices...")
  initial_global_coordinates = compute_global_coordinates(local_coordinates_init_pose, parents)
  current_global_coordinates = compute_global_coordinates(local_coordinates_current_pose, parents)
  skin_transforms = current_global_coordinates @ np.linalg.inv(initial_global_coordinates)
  # skin_transforms = T_k . (T^0_k)^-1

  joint_positions = initial_global_coordinates[:, 0:2, 2]
  vertex_indices = np.arange(skin_vertex_positions.shape[0])

  # Measure each vertex's distance to every bone segment.
  child_frames = np.flatnonzero(parents >= 0)
  parent_frames = parents[child_frames]
  segment_starts = joint_positions[parent_frames]
  segment_ends = joint_positions[child_frames]
  segments = segment_ends - segment_starts
  segment_lengths_squared = np.sum(segments * segments, axis=1)
  offsets = skin_vertex_positions[:, np.newaxis, :] - segment_starts[np.newaxis, :, :]
  projections = np.sum(offsets * segments[np.newaxis, :, :], axis=2)
  # Let u := offsets, v := segments, then projections := u.v = ||u||.||v||.cos(u,v)
  
  segment_fractions = np.divide(
      projections,
      segment_lengths_squared[np.newaxis, :],
      out=np.zeros_like(projections),
      where=segment_lengths_squared[np.newaxis, :] > 1e-24,
  )
  segment_fractions = np.clip(segment_fractions, 0.0, 1.0)
  # segment_fractions := (u.v)/(||v||^2) = ||u||.cos(u,v) / ||v||
  # ||u||.cos(u,v) is length of the segment from segment_starts to projection of u onto v

  closest_bone_points = (
      segment_starts[np.newaxis, :, :]
      + segment_fractions[:, :, np.newaxis] * segments[np.newaxis, :, :]
  )
  # closest_bone_points := the projection of u onto v

  bone_distances = np.linalg.norm(
      skin_vertex_positions[:, np.newaxis, :] - closest_bone_points,
      axis=2,
  )
  # bone_distances = length of the perpendicular line segment 
  # from skin_vertex_positions to segments
  
  nearest_bone_indices = np.argmin(bone_distances, axis=1)
  d1 = bone_distances[vertex_indices, nearest_bone_indices]
  bone_frames = parent_frames[nearest_bone_indices]

  # Also find the nearest frame for each vertex.
  frame_distances = np.linalg.norm(
      skin_vertex_positions[:, np.newaxis, :] - joint_positions[np.newaxis, :, :],
      axis=2,
  )
  nearest_frames = np.argmin(frame_distances, axis=1)
  d2 = frame_distances[vertex_indices, nearest_frames]

  # Normalize inverse-distance weights. Exact frame hits remain attached to that frame.
  frame_weights = np.ones_like(d1)
  bone_weights = np.zeros_like(d1)
  has_frame_distance = d2 > 1e-12
  inverse_bone_distance = 1.0 / np.maximum(d1[has_frame_distance], 1e-12)
  # inverse_bone_distance = d1^-1
  
  inverse_frame_distance = 1.0 / d2[has_frame_distance]
  # inverse_frame_distance = d2^-1
  
  inverse_weight_sum = inverse_bone_distance + inverse_frame_distance
  # inverse_weight_sum = d1^-1 + d2^-1
  
  bone_weights[has_frame_distance] = inverse_bone_distance / inverse_weight_sum
  # bone_weights = d1^-1/(d1^-1 + d2^-1)
  
  frame_weights[has_frame_distance] = inverse_frame_distance / inverse_weight_sum
  # frame_weights = d2^-1/(d1^-1 + d2^-1)

  weights = np.zeros(
      (skin_vertex_positions.shape[0], joint_positions.shape[0]),
      dtype=skin_vertex_positions.dtype,
  )

  weights[vertex_indices, bone_frames] += bone_weights
  weights[vertex_indices, nearest_frames] += frame_weights

  homogeneous_vertices = np.column_stack((
      skin_vertex_positions,
      np.ones(skin_vertex_positions.shape[0], dtype=skin_vertex_positions.dtype),
  ))
  # change to coordinates to the homogeneous coordinates

  transformed_vertices = np.einsum("fij,vj->vfi", skin_transforms, homogeneous_vertices)
  # skin_transforms[f,:,:] = the matrix M_f = T_f.(T0_f)^-1 of size 3x3
  # homogeneous_vertices[v, :] = 3 coordinates of the skin vertex v
  # transformed_vertices[v,f,:] = 3 new coordinates the skin vertex v after changed by frame f.

  updated_skin_vertices = np.einsum("vf,vfi->vi", weights, transformed_vertices)[:, :2]
  # weights[v,f] = weight of the frame f on the vertex v
  # transformed_vertices[v,f,:] = as above
  # updated_skin_vertices[v, :] = 3 coordinates equal to 
  # sum of the transformed_vertices[v,f,:]*weights[v,f] over all f
  return updated_skin_vertices

# add white background to the scene
def addBackground():
  # background is needed for world_pos to be properly defined when mouse clicking on a point
  vertices = np.array([[-10,-10],[10,-10],[10,10],[-10,10]])
  indices = np.array([[0,1,2,3]])
  ps_bg_plane = ps.register_surface_mesh("background", vertices, indices)
  ps_bg_plane.set_color((1, 1, 1))

if __name__=="__main__":
  # Define skeleton
  nframes = 8
  local_coordinates = np.zeros((nframes,3,3), np.float64)
  local_coordinates[0] = np.array([[1,0,0],[0,1,0],[0,0,1]], np.float64) #upper arm
  local_coordinates[1] = np.array([[1,0,0.5],[0,1,0],[0,0,1]], np.float64) #form arm
  local_coordinates[2] = np.array([[1,0,0.5],[0,1,0],[0,0,1]], np.float64) #hand
  #fingers:
  local_coordinates[3] = np.array([[1,0,0.03],[0,1,0.12],[0,0,1]], np.float64)
  local_coordinates[4] = np.array([[1,0,0.2],[0,1,0.08],[0,0,1]], np.float64)
  local_coordinates[5] = np.array([[1,0,0.23],[0,1,0],[0,0,1]], np.float64)
  local_coordinates[6] = np.array([[1,0,0.2],[0,1,-0.08],[0,0,1]], np.float64)
  local_coordinates[7] = np.array([[1,0,0.15],[0,1,-0.12],[0,0,1]], np.float64)
  local_coordinates_init = np.copy(local_coordinates)
  # Define parent tree (i.e. skeleton structure)
  parents = np.array([-1,0,1,2,2,2,2,2], np.int32)
    
  global_coordinates = compute_global_coordinates(local_coordinates, parents)
  # Get end positions of the bones (world coordinates)
  joint_positions = global_coordinates[:,0:2,2]
  # Define bones edges (for visualization purpose)
  bones = np.array([[0,1],[1,2],[2,3],[2,4],[2,5],[2,6],[2,7]])
  
  # Define skin vertices (world coordinates)
  n = 10
  nvertex = 2*n+9
  skin_vertex_positions = np.zeros((nvertex,2), np.float64)
  # top part of the upper and lower arm:
  x = np.linspace(0,0.98,n)
  skin_vertex_positions[0:n, :] = np.column_stack((x,0.08-0.06*x))
  # fingers:
  skin_vertex_positions[n, :] = joint_positions[3, :]
  skin_vertex_positions[n+1, :] = [1.036, 0.046]
  skin_vertex_positions[n+2, :] = joint_positions[4, :]
  skin_vertex_positions[n+3, :] = [1.11, 0.019]
  skin_vertex_positions[n+4, :] = joint_positions[5, :]
  skin_vertex_positions[n+5, :] = [1.104, -0.022]
  skin_vertex_positions[n+6, :] = joint_positions[6, :]
  skin_vertex_positions[n+7, :] = [1.094, -0.053]
  skin_vertex_positions[n+8, :] = joint_positions[7, :]
  skin_vertex_positions[n+9:2*n+9, :] = np.column_stack((np.flip(x),-(0.08-0.06*np.flip(x))))
  skin_vertex_positions_init = np.copy(skin_vertex_positions)
  # Define skin edges (for visualization purpose)
  skin_edges = np.column_stack((np.linspace(0,nvertex-1,nvertex).astype(int), np.linspace(1,nvertex,nvertex).astype(int)))
  skin_edges[nvertex-1,1] = 0

  # Set up UI
  ui_options = ["forward kinematics", "inverse kinematics"]
  ui_options_selected = ui_options[0]
  show_skin = False
  ui_selected_bone_index = 0

  def callback():
    # Executed every frame
    global ui_options_selected
    option_changed = psim.BeginCombo("Pick one", ui_options_selected)
    if option_changed:
        for val in ui_options:
            selected = psim.Selectable(val, ui_options_selected==val)
            if selected:
                ui_options_selected = val
        psim.EndCombo()
      
    global ui_selected_bone_index
    bone_changed, ui_selected_bone_index = psim.SliderInt("selected bone (0, 1 or 2)", ui_selected_bone_index, v_min=0, v_max=2)
    
    global show_skin
    show_skin_changed, show_skin = psim.Checkbox("show skin", show_skin)
    #print("show_skin_changed0", show_skin_changed)
    if show_skin_changed:
        ps_skin_v.set_enabled(show_skin)
        ps_skin_e.set_enabled(show_skin)
        
    io = psim.GetIO()
    if io.MouseClicked[0] and io.KeyCtrl:
      print("show_skin_changed", show_skin_changed)
      screen_coords = io.MousePos
      world_pos = ps.screen_coords_to_world_position(screen_coords)
      print("  world pos = ", world_pos)
      
      global local_coordinates, local_coordinates_init, parents
      global global_coordinates, joint_positions
      global skin_vertex_positions
      
      if ui_options_selected=="forward kinematics":
        local_coordinates = update_skeleton_using_forward_kinematics(local_coordinates, parents, ui_selected_bone_index, world_pos)
      elif ui_options_selected=="inverse kinematics":
        local_coordinates = update_skeleton_using_FABRIC(local_coordinates, parents, world_pos)
      
      if show_skin:
        skin_vertex_positions = update_skin_vertices(skin_vertex_positions_init, local_coordinates_init, local_coordinates, parents)
        ps_skin_v.update_point_positions(skin_vertex_positions)
        ps_skin_e.update_node_positions(skin_vertex_positions)
      
      global_coordinates = compute_global_coordinates(local_coordinates, parents)
      joint_positions = global_coordinates[:,0:2,2]
      ps_joints.update_point_positions(joint_positions)
      ps_bones.update_node_positions(joint_positions)
     
     
  # Initialize Polyscope
  ps.init()
  ps.set_navigation_style("planar")
  
  # adjust bounding box so that the object is properly centered
  ps.set_automatically_compute_scene_extents(False)
  ps.set_length_scale(1.)
  low = np.array((-1, -1.5, -1.))
  high = np.array((1.5, 1.5, 1.))
  ps.set_bounding_box(low, high)
  
  ps_joints = ps.register_point_cloud("joint positions", joint_positions, radius=0.01)
  ps_bones = ps.register_curve_network("bones", joint_positions, bones)
  ps_skin_v = ps.register_point_cloud("skin vertex positions", skin_vertex_positions, radius=0.006)
  ps_skin_e = ps.register_curve_network("skin edges", skin_vertex_positions, skin_edges, radius=0.004)
  ps_skin_v.set_enabled(show_skin)
  ps_skin_e.set_enabled(show_skin)
  
  addBackground()
  
  ps.set_user_callback(callback)
  
  ps.show()
