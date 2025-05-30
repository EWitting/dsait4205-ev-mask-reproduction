import numpy as np
import statistics
import math
import cv2

# --- generate_event_arrays, gen_extremities, generate_cropped_frames (Keep these as they are) ---
# (Your existing implementations of these functions are fine and will be used)
def generate_event_arrays(events, polarity, div_rate=300, mult_rate=100):
    filtered_events = list(filter(lambda entry: entry[3] == polarity, events))
    x_data = [entry[0] for entry in filtered_events]
    y_data = [entry[1] for entry in filtered_events]
    z_data = [int(i / div_rate) * mult_rate for i, entry in enumerate(filtered_events)]
    # time_data is now in MICROSECONDS, which is better for tau
    time_data = [int(entry[2]) for entry in filtered_events] # Keep as microseconds
    return np.array(x_data), np.array(y_data), np.array(z_data), np.array(time_data)


def gen_extremities(x_data_pos, y_data_pos, x_data_neg, y_data_neg,
                    current_i, i, current_j, j):
    # Note: This function might need adjustment if event arrays can be empty for a window
    # For simplicity, assuming it handles empty slices gracefully or they are filtered out before calling.
    
    all_x_coords = []
    all_y_coords = []

    if i > current_i and len(x_data_pos) > 0 : # Check if there are positive events to slice
        all_x_coords.extend(x_data_pos[current_i:i])
        all_y_coords.extend(y_data_pos[current_i:i])
    
    if j > current_j and len(x_data_neg) > 0: # Check if there are negative events to slice
        all_x_coords.extend(x_data_neg[current_j:j])
        all_y_coords.extend(y_data_neg[current_j:j])

    if not all_x_coords: # No events in this combined slice
        return 0, 0, 0, 0 # len_x, len_y, overall_x, overall_y

    neg_mn_x = min(all_x_coords)
    neg_mn_y = min(all_y_coords)
    neg_mx_x = max(all_x_coords)
    neg_mx_y = max(all_y_coords)

    len_x = neg_mx_x - neg_mn_x
    len_y = neg_mx_y - neg_mn_y

    overall_x = int(((neg_mx_x - neg_mn_x) / 2) + neg_mn_x)
    overall_y = int(((neg_mx_y - neg_mn_y) / 2) + neg_mn_y)

    return len_x, len_y, overall_x, overall_y


def generate_cropped_frames(len_arr_x, len_arr_y, frames_list, center_indices_list):
    if not len_arr_x or not len_arr_y: # Handle empty inputs
        return [], 0, 0, []

    # Filter out zero lengths if any, to avoid issues with statistics.mean
    valid_len_x = [lx for lx in len_arr_x if lx > 0]
    valid_len_y = [ly for ly in len_arr_y if ly > 0]

    if not valid_len_x or not valid_len_y: # If all lengths were zero or empty
         hx, hy = 10, 10 # Default crop size if no valid event extents
    else:
        mean_len_x = statistics.mean(valid_len_x)
        mean_len_y = statistics.mean(valid_len_y)
        hx, hy = math.ceil(mean_len_x / 2), math.ceil(mean_len_y / 2)
        # Ensure hx, hy are at least 1 to avoid zero-size crops
        hx = max(1, hx)
        hy = max(1, hy)


    cropped_frames_list, cropping_positions_list = [], []
    for ind, frame_item in enumerate(frames_list):
        (cx, cy) = center_indices_list[ind]
        # Ensure cx, cy are within frame bounds if possible
        # Though gen_extremities should give valid centers based on event data
        
        # Calculate crop coordinates, ensuring they are within frame dimensions
        y0 = max(0, cy - hy)
        y1 = min(frame_item.shape[0], cy + hy + 1) # +1 because slicing is exclusive at end
        x0 = max(0, cx - hx)
        x1 = min(frame_item.shape[1], cx + hx + 1)

        # Ensure y1 > y0 and x1 > x0 to prevent empty slices
        if y1 <= y0: y1 = y0 + 1
        if x1 <= x0: x1 = x0 + 1
        
        # Final check to ensure crop is within frame boundaries after adjustments
        y1 = min(frame_item.shape[0], y1)
        x1 = min(frame_item.shape[1], x1)


        cropped = frame_item[y0:y1, x0:x1]
        if cropped.size == 0: # If crop somehow becomes empty
            # Fallback: use a small default crop or the full frame
            # print(f"Warning: Empty crop for frame {ind}. Using fallback.")
            cropped = frame_item[0:min(10, frame_item.shape[0]), 0:min(10, frame_item.shape[1])] # Example small crop
            if cropped.size == 0: # If frame itself is too small
                cropped = np.zeros((1,1,frame_item.shape[2] if len(frame_item.shape) == 3 else 1), dtype=frame_item.dtype)


        cropped_frames_list.append(cropped)
        cropping_positions_list.append((y0, y1, x0, x1)) # Store as y0,y1,x0,x1

    return cropped_frames_list, hx, hy, cropping_positions_list


# --- Helper function for Time Surface generation (internal to this module) ---
def _create_single_time_surface(event_xs, event_ys, event_ts,
                                t_end_window, tau, img_height, img_width):
    surface = np.full((img_height, img_width), -np.inf, dtype=np.float64)
    
    if event_ts.size == 0: # No events of this polarity
        return np.zeros((img_height, img_width), dtype=np.float32)

    for i in range(event_ts.size):
        x, y, t = int(event_xs[i]), int(event_ys[i]), event_ts[i]
        if 0 <= y < img_height and 0 <= x < img_width:
            surface[y, x] = max(surface[y, x], t)
            
    dt = t_end_window - surface
    exp_surface = np.exp(-dt / tau)
    exp_surface[np.isinf(surface)] = 0 # Pixels with no event become 0
    
    # Normalize to [0, 1]
    min_val, max_val = np.min(exp_surface), np.max(exp_surface)
    if max_val > min_val:
        exp_surface = (exp_surface - min_val) / (max_val - min_val)
    else: # All same value (likely all zeros if no events or tau is tiny)
        exp_surface = np.zeros_like(exp_surface, dtype=np.float32)
        
    return exp_surface.astype(np.float32)


# --- Modified generate_event_frames_with_fixed_time_window ---
def generate_event_frames_with_fixed_time_window(
    positive_event_array_denoised, negative_event_array_denoised,
    positive_event_array, negative_event_array, # Noisy events
    window_len=20000, img_shape=(34, 34), # window_len in MICROSECONDS
    representation_mode="rgbd", # "rgbd" or "ts"
    tau_on=10000, tau_off=10000, # MICROSECONDS, for TS mode
    **kwargs
    ):

    print(f"Generating frames with window_len={window_len}us, representation_mode={representation_mode}")
    img_height, img_width = img_shape

    # Unpack DENOISED event arrays (timestamps are in MICROSECONDS)
    x_data_pos_den, y_data_pos_den, _, time_data_pos_den = positive_event_array_denoised
    x_data_neg_den, y_data_neg_den, _, time_data_neg_den = negative_event_array_denoised

    # Unpack NOISY event arrays (timestamps are in MICROSECONDS)
    x_data_pos, y_data_pos, _, time_data_pos = positive_event_array
    x_data_neg, y_data_neg, _, time_data_neg = negative_event_array

    # Output lists
    output_main_frames = []       # List of 3-channel (H,W,3) uint8 frames
    output_denoised_polarity_frames = [] # List of 3-channel (H,W,3) uint8 denoised polarity frames (for cropping)
    output_depth_compatible_frames = [] # List of 1-channel (H,W,1) uint8 "depth" frames

    len_arr_x_list, len_arr_y_list, center_indices_list = [], [], []
    
    # Pointers for iterating through DENOISED event arrays
    idx_den_pos, idx_den_neg = 0, 0
    # Pointers for iterating through NOISY event arrays (for RGBD mode)
    idx_noisy_pos, idx_noisy_neg = 0, 0

    # Determine the overall time range from DENOISED events to define windows
    all_denoised_times = []
    if time_data_pos_den.size > 0: all_denoised_times.extend(time_data_pos_den)
    if time_data_neg_den.size > 0: all_denoised_times.extend(time_data_neg_den)

    if not all_denoised_times: # No denoised events to define windows
        # print("Warning: No denoised events available to define time windows.")
        return [], [], [], 0, 0, [], []

    min_overall_time = np.min(all_denoised_times)
    max_overall_time = np.max(all_denoised_times)
    
    current_window_start_time = min_overall_time

    while current_window_start_time <= max_overall_time:
        current_window_end_time = current_window_start_time + window_len

        # --- 1. Prepare DENOISED polarity frame for this window (for cropping/extremities) ---
        current_denoised_polarity_frame = np.zeros((img_height, img_width, 3), np.uint8)
        
        # Store indices for gen_extremities for denoised events
        start_idx_den_pos_win, end_idx_den_pos_win = idx_den_pos, idx_den_pos
        start_idx_den_neg_win, end_idx_den_neg_win = idx_den_neg, idx_den_neg

        # Accumulate DENOISED positive events for current_denoised_polarity_frame
        while end_idx_den_pos_win < time_data_pos_den.size and \
              time_data_pos_den[end_idx_den_pos_win] < current_window_end_time:
            if time_data_pos_den[end_idx_den_pos_win] >= current_window_start_time:
                x, y = int(x_data_pos_den[end_idx_den_pos_win]), int(y_data_pos_den[end_idx_den_pos_win])
                if 0 <= y < img_height and 0 <= x < img_width:
                    current_denoised_polarity_frame[y, x] = (255, 0, 0)  # Blue (ON)
            end_idx_den_pos_win += 1
        
        # Accumulate DENOISED negative events
        while end_idx_den_neg_win < time_data_neg_den.size and \
              time_data_neg_den[end_idx_den_neg_win] < current_window_end_time:
            if time_data_neg_den[end_idx_den_neg_win] >= current_window_start_time:
                x, y = int(x_data_neg_den[end_idx_den_neg_win]), int(y_data_neg_den[end_idx_den_neg_win])
                if 0 <= y < img_height and 0 <= x < img_width:
                    current_denoised_polarity_frame[y, x] = (0, 0, 255)  # Red (OFF)
            end_idx_den_neg_win += 1
        
        # Count actual events processed for the denoised polarity frame for this window
        num_den_pos_events_in_win = np.count_nonzero( (time_data_pos_den[idx_den_pos:end_idx_den_pos_win] >= current_window_start_time) )
        num_den_neg_events_in_win = np.count_nonzero( (time_data_neg_den[idx_den_neg:end_idx_den_neg_win] >= current_window_start_time) )


        # --- 2. Generate MAIN output frame and DEPTH frame based on mode ---
        main_frame_for_this_window = np.zeros((img_height, img_width, 3), np.uint8)
        depth_frame_for_this_window = np.zeros((img_height, img_width, 1), np.uint8) # Ensure single channel

        if representation_mode == "ts":
            # Collect DENOISED events for TS generation
            ts_pos_xs = x_data_pos_den[idx_den_pos:end_idx_den_pos_win][time_data_pos_den[idx_den_pos:end_idx_den_pos_win] >= current_window_start_time]
            ts_pos_ys = y_data_pos_den[idx_den_pos:end_idx_den_pos_win][time_data_pos_den[idx_den_pos:end_idx_den_pos_win] >= current_window_start_time]
            ts_pos_ts = time_data_pos_den[idx_den_pos:end_idx_den_pos_win][time_data_pos_den[idx_den_pos:end_idx_den_pos_win] >= current_window_start_time]

            ts_neg_xs = x_data_neg_den[idx_den_neg:end_idx_den_neg_win][time_data_neg_den[idx_den_neg:end_idx_den_neg_win] >= current_window_start_time]
            ts_neg_ys = y_data_neg_den[idx_den_neg:end_idx_den_neg_win][time_data_neg_den[idx_den_neg:end_idx_den_neg_win] >= current_window_start_time]
            ts_neg_ts = time_data_neg_den[idx_den_neg:end_idx_den_neg_win][time_data_neg_den[idx_den_neg:end_idx_den_neg_win] >= current_window_start_time]

            ts_on_surface = _create_single_time_surface(ts_pos_xs, ts_pos_ys, ts_pos_ts,
                                                        current_window_end_time, tau_on, img_height, img_width)
            ts_off_surface = _create_single_time_surface(ts_neg_xs, ts_neg_ys, ts_neg_ts,
                                                         current_window_end_time, tau_off, img_height, img_width)
            
            main_frame_for_this_window[:, :, 0] = (ts_on_surface * 255).astype(np.uint8)  # Blue = ON-TS
            main_frame_for_this_window[:, :, 2] = (ts_off_surface * 255).astype(np.uint8) # Red  = OFF-TS
            # depth_frame_for_this_window remains zeros as requested

        elif representation_mode == "rgbd":
            # Use NOISY events for the main RGBD polarity frame and depth/recency frame
            start_idx_noisy_pos_win = idx_noisy_pos
            start_idx_noisy_neg_win = idx_noisy_neg

            # Noisy Positive Events
            while idx_noisy_pos < time_data_pos.size and time_data_pos[idx_noisy_pos] < current_window_end_time:
                if time_data_pos[idx_noisy_pos] >= current_window_start_time:
                    x, y = int(x_data_pos[idx_noisy_pos]), int(y_data_pos[idx_noisy_pos])
                    if 0 <= y < img_height and 0 <= x < img_width:
                        main_frame_for_this_window[y, x] = (255, 0, 0)  # Blue (ON)
                        depth_frame_for_this_window[y, x, 0] = (idx_noisy_pos - start_idx_noisy_pos_win + 1) # Recency
                idx_noisy_pos += 1
            
            # Noisy Negative Events
            while idx_noisy_neg < time_data_neg.size and time_data_neg[idx_noisy_neg] < current_window_end_time:
                if time_data_neg[idx_noisy_neg] >= current_window_start_time:
                    x, y = int(x_data_neg[idx_noisy_neg]), int(y_data_neg[idx_noisy_neg])
                    if 0 <= y < img_height and 0 <= x < img_width:
                        main_frame_for_this_window[y, x] = (0, 0, 255)  # Red (OFF)
                        depth_frame_for_this_window[y, x, 0] = (idx_noisy_neg - start_idx_noisy_neg_win + 1) # Recency
                idx_noisy_neg += 1
            
            # Histogram Equalization for depth frame (original logic)
            if np.count_nonzero(depth_frame_for_this_window) > 0:
                # Squeeze, equalize, then reshape back to (H,W,1)
                squeezed_depth = depth_frame_for_this_window.squeeze().astype(np.uint8)
                equalized_hist_depth = cv2.equalizeHist(squeezed_depth)
                
                x_pixels, y_pixels = np.where(equalized_hist_depth > 0)
                scaled_values = equalized_hist_depth[x_pixels, y_pixels] / 2 + 50
                equalized_hist_depth[x_pixels, y_pixels] = np.clip(scaled_values, 0, 255).astype(np.uint8)
                depth_frame_for_this_window = equalized_hist_depth.reshape((img_height, img_width, 1))
        else:
            raise ValueError(f"Unknown representation_mode: {representation_mode}")

        # --- 3. Condition for saving the generated frame ---
        # Original condition: np.count_nonzero(current_frame) > 100 AND np.count_nonzero(current_frame_den) > 0
        # AND i_den > current_i_den AND j_den > current_j_den
        # New interpretation: >100 events in main output frame, >0 events in denoised polarity, and some denoised events processed.
        
        processed_any_denoised_in_win = (num_den_pos_events_in_win > 0 or num_den_neg_events_in_win > 0)

        if np.count_nonzero(main_frame_for_this_window) > 100 and \
           np.count_nonzero(current_denoised_polarity_frame) > 0 and \
           processed_any_denoised_in_win:
            
            # Call gen_extremities with indices relative to the start of the current window's DENOISED events
            # The actual arrays x_data_pos_den etc. are passed, with slices defined by current/end indices.
            # Note: Original gen_extremities took slices from start of arrays; now we need to be careful.
            # It's safer to pass the actual event coordinates from the current window.
            
            # For gen_extremities, we pass the start and end pointers for DENOISED events for *this window*
            # These pointers (idx_den_pos, end_idx_den_pos_win) are for the full denoised arrays.
            len_x, len_y, overall_x, overall_y = gen_extremities(
                x_data_pos_den, y_data_pos_den, x_data_neg_den, y_data_neg_den,
                idx_den_pos, end_idx_den_pos_win,  # Range of positive denoised events considered for this window
                idx_den_neg, end_idx_den_neg_win   # Range of negative denoised events
            )

            output_main_frames.append(main_frame_for_this_window)
            output_denoised_polarity_frames.append(current_denoised_polarity_frame)
            output_depth_compatible_frames.append(depth_frame_for_this_window)
            
            len_arr_x_list.append(len_x)
            len_arr_y_list.append(len_y)
            center_indices_list.append((overall_x, overall_y))

        # Advance DENOISED event pointers to the start of the next potential window
        idx_den_pos = end_idx_den_pos_win
        idx_den_neg = end_idx_den_neg_win
        
        current_window_start_time += window_len # Move to the next processing window

    if not output_main_frames:
        return [], [], [], 0, 0, [], []

    # --- 4. Cropping (uses output_denoised_polarity_frames for consistent cropping) ---
    cropped_output_frames, hx_crop, hy_crop, cropping_positions_list = \
        generate_cropped_frames(len_arr_x_list, len_arr_y_list, output_denoised_polarity_frames, center_indices_list)

    # Return structure:
    # 1. frames (main output: RGBD polarity OR TS-based)
    # 2. frames_denoised (always polarity-based, used for mask placement heuristics by caller)
    # 3. cropped_frames (cropped versions of frames_denoised)
    # 4. hx * 2 + 1 (crop width dim)
    # 5. hy * 2 + 1 (crop height dim)
    # 6. cropping_positions
    # 7. time_frames (depth: recency for RGBD OR zeros for TS)
    return output_main_frames, output_denoised_polarity_frames, cropped_output_frames, \
           hx_crop * 2 + 1, hy_crop * 2 + 1, cropping_positions_list, output_depth_compatible_frames


# --- generate_fixed_num_events_frames (Keep as is, or adapt similarly if needed) ---
# This function has a different windowing logic (based on num events, not time)
# If you also want to adapt this one for TS, it would require a similar pattern of changes.
# For now, I'm leaving it as per your original file.
def generate_fixed_num_events_frames(positive_event_array, negative_event_array, total_frames=20, img_shape=(34, 34)):
    img_height, img_width = img_shape
    x_data_pos, y_data_pos, z_data_pos, time_data_pos = positive_event_array
    x_data_neg, y_data_neg, z_data_neg, time_data_neg = negative_event_array

    # Handle cases where event arrays might be empty or too small
    if len(time_data_pos) == 0 or len(time_data_neg) == 0 or total_frames == 0:
        return [], [], 0, 0, [], [] # Return empty if no data or no frames to make
    
    window_len_pos = max(1, int(len(time_data_pos) / total_frames)) # Ensure at least 1 event
    window_len_neg = max(1, int(len(time_data_neg) / total_frames))

    frames, len_arr_x, len_arr_y, center_indices, time_frames_list = [], [], [], [], []
    i, j = 0, 0

    while i < len(time_data_pos) and j < len(time_data_neg):
        current_i, current_j = i, j
        current_frame = np.zeros((img_height, img_width, 3), np.uint8)
        time_frame_single = np.zeros((img_height, img_width, 1), np.uint8)

        # Positive events for this frame
        end_i = min(len(time_data_pos), current_i + window_len_pos)
        for k_pos in range(current_i, end_i):
            x, y = int(x_data_pos[k_pos]), int(y_data_pos[k_pos])
            if 0 <= y < img_height and 0 <= x < img_width:
                current_frame[y, x] = (255, 0, 0)   # Blue
                time_frame_single[y, x, 0] = (k_pos - current_i + 1) # Recency
        i = end_i
        
        # Negative events for this frame
        end_j = min(len(time_data_neg), current_j + window_len_neg)
        for k_neg in range(current_j, end_j):
            x, y = int(x_data_neg[k_neg]), int(y_data_neg[k_neg])
            if 0 <= y < img_height and 0 <= x < img_width:
                current_frame[y, x] = (0, 0, 255)   # Red
                time_frame_single[y, x, 0] = max(time_frame_single[y,x,0], (k_neg - current_j + 1)) # Take max recency
        j = end_j
        
        if np.count_nonzero(current_frame) > 50:
            equalized_hist = cv2.equalizeHist(time_frame_single.squeeze().astype(np.uint8))
            # Original paper's scaling, if desired (apply carefully after equalization)
            # x_pixels, y_pixels = np.where(equalized_hist > 0)
            # scaled_values = equalized_hist[x_pixels, y_pixels] / 2 + 50
            # equalized_hist[x_pixels, y_pixels] = np.clip(scaled_values, 0, 255).astype(np.uint8)
            
            # gen_extremities needs start/end indices for the *current window*
            len_x, len_y, overall_x, overall_y = gen_extremities(
                x_data_pos, y_data_pos, x_data_neg, y_data_neg, 
                current_i, i, current_j, j
            )
            frames.append(current_frame)
            len_arr_x.append(len_x)
            len_arr_y.append(len_y)
            center_indices.append((overall_x, overall_y))
            time_frames_list.append(equalized_hist.reshape(img_height,img_width,1)) # Reshape back

    if not frames:
        return [], [], 0, 0, [], []
        
    cropped_frames_list, hx, hy, cropping_positions_list = \
        generate_cropped_frames(len_arr_x, len_arr_y, frames, center_indices)

    return frames, cropped_frames_list, hx * 2 + 1, hy * 2 + 1, cropping_positions_list, time_frames_list































# import numpy as np
# import statistics
# import math
# import cv2

# def generate_event_arrays(events, polarity, div_rate=300, mult_rate=100):
#     """
#     Used for splitting the array of events into 3 arrays based on polarity (=sign of the event).

#     Input:
#     events = stream of events, one entry looks like: (x, y, time, polarity)
#     polarity = 1 or 0, 1 for generating array of positive events, 0 for generating array of negative events

#     Output (same indices for same event):
#     x_data = array with position on the x-axis of the events
#     y_data = array with position on the y-axis of the events
#     z_data = time of the event (can be flattened with div/mult_rate) -> set both to 1 for no flattening
#     """
#     filtered_events = list(filter(lambda entry: entry[3] == polarity, events))

#     x_data = [entry[0] for entry in filtered_events]
#     y_data = [entry[1] for entry in filtered_events]
#     z_data = [int(i / div_rate) * mult_rate for i, entry in enumerate(filtered_events)]
#     time_data = [int(entry[2]/1000) for entry in filtered_events] # Convert time from microseconds to miliseconds (ms)

#     return x_data, y_data, z_data, time_data


# def generate_event_frames(positive_event_array, negative_event_array, window_len=10, img_shape=(34, 34)):
#     """
#     Takes events in intervals of len=window_len and turns them into a frame, blue=positive, red=negative.
#     :param positive_event_array: Positive events (x, y, custom_z, timestamp_in_ms).
#     :param negative_event_array: Negative events (x, y, custom_z, timestamp_in_ms).
#     :param window_len: Len of time interval that should be allowed in the same frame.
#     :param img_shape: How big the output frames should be.
#     :return: Array with frames created from event stream.
#     """

#     img_height, img_width = img_shape
#     frames = []
#     i, j = 0, 0

#     x_data_pos, y_data_pos, z_data_pos, time_data_pos = positive_event_array
#     x_data_neg, y_data_neg, z_data_neg, time_data_neg = negative_event_array

#     while i < len(time_data_pos) and j < len(time_data_neg):
#         current_time = min(time_data_pos[i], time_data_neg[j])
#         current_frame = np.zeros((img_height, img_width, 3), np.uint8)
#         # current_frame.fill(255)
#         while i < len(time_data_pos) and time_data_pos[i] < current_time + window_len:
#             x = x_data_pos[i]
#             y = y_data_pos[i]
#             current_frame[y][x] = (255, 0, 0)   # Blue
#             i += 1
#         while j < len(time_data_neg) and time_data_neg[j] < current_time + window_len:
#             x = x_data_neg[j]
#             y = y_data_neg[j]
#             current_frame[y][x] = (0, 0, 255)   # Red
#             j += 1
#         frames.append(current_frame)

#     return frames


# def gen_extremities(x_data_pos, y_data_pos, x_data_neg, y_data_neg,
#                     current_i, i, current_j, j):
#     """
#     Figures out the edges (min, max extremities) of a frame based on event positions.
#     :param x_data_pos:
#     :param y_data_pos:
#     :param x_data_neg:
#     :param y_data_neg:
#     :param current_i:
#     :param i:
#     :param current_j:
#     :param j:
#     :return:
#     """
#     # pos_mn_x = min(x_data_pos[current_i:i])
#     # pos_mn_y = min(y_data_pos[current_i:i])
#     # pos_mx_x = max(x_data_pos[current_i:i])
#     # pos_mx_y = max(y_data_pos[current_i:i])

#     # pos_overall_x = ((pos_mx_x - pos_mn_x) / 2) + pos_mn_x
#     # pos_overall_y = ((pos_mx_y - pos_mn_y) / 2) + pos_mn_y

#     neg_mn_x = min(x_data_neg[current_j:j])
#     neg_mn_y = min(y_data_neg[current_j:j])
#     neg_mx_x = max(x_data_neg[current_j:j])
#     neg_mx_y = max(y_data_neg[current_j:j])

#     # len_x = max(pos_mx_x, neg_mx_x) - min(pos_mn_x, neg_mn_x)
#     # len_y = max(pos_mx_y, neg_mx_y) - min(pos_mn_y, neg_mn_y)

#     len_x = neg_mx_x - neg_mn_x
#     len_y = neg_mx_y - neg_mn_y

#     neg_overall_x = ((neg_mx_x - neg_mn_x) / 2) + neg_mn_x
#     neg_overall_y = ((neg_mx_y - neg_mn_y) / 2) + neg_mn_y

#     # overall_x = int((pos_overall_x + neg_overall_x) / 2)
#     # overall_y = int((pos_overall_y + neg_overall_y) / 2)
#     overall_x = int(neg_overall_x)
#     overall_y = int(neg_overall_y)

#     return len_x, len_y, overall_x, overall_y


# def generate_cropped_frames(len_arr_x, len_arr_y, frames, center_indices):
#     """
#     Creates the cropped frames, their positions and the average centers, as explained in 'generate_fixed_num_events_frames'.
#     :param len_arr_x:
#     :param len_arr_y:
#     :param frames:
#     :param center_indices:
#     :return:
#     """
#     if len(len_arr_x) > 0 and len(len_arr_y) > 0:
#         mean_len_x = statistics.mean(len_arr_x)
#         mean_len_y = statistics.mean(len_arr_y)
#         hx, hy = math.ceil(mean_len_x / 2), math.ceil(mean_len_y / 2)
#     else:
#         hx, hy = 0, 0

#     cropped_frames, cropping_positions = [], []
#     for ind, frame in enumerate(frames):
#         (cx, cy) = center_indices[ind]
#         x0, x1 = max(0, cx - hx), min(frame.shape[1], cx + hx)
#         y0, y1 = max(0, cy - hy), min(frame.shape[1], cy + hy)
#         # # Add purple corners
#         # print(cx, cy, hx, hy)
#         # frame[y0][x0] = (255, 0, 255)
#         # frame[y0][x1] = (255, 0, 255)
#         # frame[y1][x0] = (255, 0, 255)
#         # frame[y1][x1] = (255, 0, 255)

#         cropped = frame[y0: y1 + 1, x0: x1 + 1]
#         cropped_frames.append(cropped)
#         cropping_positions.append((y0, y1 + 1, x0, x1 + 1))
#         # cv2.imshow('frame', frame)
#         # cv2.waitKey(500)

#     return cropped_frames, hx, hy, cropping_positions


# def generate_event_frames_with_fixed_time_window(positive_event_array_denoised, negative_event_array_denoised,
#                                                  positive_event_array, negative_event_array,
#                                                  window_len=20, img_shape=(34, 34)):
#     img_height, img_width = img_shape

#     x_data_pos_den, y_data_pos_den, _, time_data_pos_den = positive_event_array_denoised
#     x_data_neg_den, y_data_neg_den, _, time_data_neg_den = negative_event_array_denoised

#     x_data_pos, y_data_pos, _, time_data_pos = positive_event_array
#     x_data_neg, y_data_neg, _, time_data_neg = negative_event_array

#     frames, len_arr_x, len_arr_y, center_indices, time_frames, frames_denoised = [], [], [], [], [], []
#     i_den, j_den, i, j = 0, 0, 0, 0

#     while i_den < len(time_data_pos_den) and j_den < len(time_data_neg_den):
#         current_i_den, current_j_den, current_i, current_j = i_den, j_den, i, j
#         time_frame = np.zeros((img_height, img_width, 1), np.uint8)     # Only need the one WITH noise - DONE
#         current_time = min(time_data_pos_den[i_den], time_data_neg_den[j_den])
#         current_frame = np.zeros((img_height, img_width, 3), np.uint8)  # Only need the one WITH noise - DONE
#         current_frame_den = np.zeros((img_height, img_width, 3), np.uint8)

#         while i_den < len(time_data_pos_den) and time_data_pos_den[i_den] < current_time + window_len:
#             x = x_data_pos_den[i_den]
#             y = y_data_pos_den[i_den]
#             current_frame_den[y][x] = (255, 0, 0)  # Blue
#             i_den += 1
#         while j_den < len(time_data_neg_den) and time_data_neg_den[j_den] < current_time + window_len:
#             x = x_data_neg_den[j_den]
#             y = y_data_neg_den[j_den]
#             current_frame_den[y][x] = (0, 0, 255)  # Red
#             j_den += 1

#         # Time noised
#         while i < len(time_data_pos) and time_data_pos[i] < current_time + window_len:
#             x = x_data_pos[i]
#             y = y_data_pos[i]
#             current_frame[y][x] = (255, 0, 0)   # Blue
#             time_frame[y][x] = i - current_i  # Latest pixel gets saved
#             i += 1
#         while j < len(time_data_neg) and time_data_neg[j] < current_time + window_len:
#             x = x_data_neg[j]
#             y = y_data_neg[j]
#             current_frame[y][x] = (0, 0, 255)   # Red
#             time_frame[y][x] = j - current_j  # Latest pixel gets saved
#             j += 1

#         equalized_hist = np.array(cv2.equalizeHist(time_frame))

#         x_pixels, y_pixels = np.where(equalized_hist > 0)

#         # Because I don't want event pixels that got triggered in that interval to be 0
#         equalized_hist[x_pixels, y_pixels] = equalized_hist[x_pixels, y_pixels] / 2 + 50

#         # print(equalized_hist[white_pixels[0]:white_pixels[1]])
#          #= equalized_hist[white_pixels] / 2 + 127
#         # print()

#         # print('nonzero:', np.count_nonzero(current_frame))
#         # print(current_j, j)
#         # print(current_j_den, j_den)

#         if np.count_nonzero(current_frame) > 100 and np.count_nonzero(current_frame_den) > 0 \
#                 and current_frame.shape[0] > 0 and current_frame.shape[1] > 0 \
#                 and i_den > current_i_den and j_den > current_j_den:
#             len_x, len_y, overall_x, overall_y = \
#                 gen_extremities(x_data_pos_den, y_data_pos_den, x_data_neg_den, y_data_neg_den,
#                                 current_i_den, i_den, current_j_den, j_den)
#             frames.append(current_frame)
#             frames_denoised.append(current_frame_den)
#             len_arr_x.append(len_x)
#             len_arr_y.append(len_y)
#             center_indices.append((overall_x, overall_y))
#             time_frames.append(equalized_hist)

#             # cv2.imshow('equalized_hist', equalized_hist)
#             # cv2.imshow('current_frame', current_frame)
#             # cv2.imshow('current_frame_den', current_frame_den)
#             # cv2.imshow('equalized', equalized_hist)
#             # cv2.imshow('clahe_frame', clahe_frame)
#             # cv2.waitKey(500)

#     cropped_frames, hx, hy, cropping_positions = \
#         generate_cropped_frames(len_arr_x, len_arr_y, frames_denoised, center_indices)   # Only need the one DE-noised

#     return frames, frames_denoised, cropped_frames, hx * 2 + 1, hy * 2 + 1, cropping_positions, time_frames


# def generate_fixed_num_events_frames(positive_event_array, negative_event_array, total_frames=20, img_shape=(34, 34)):
#     """
#     Generates event frames with varying amount of events based on the total number of frames.
#     Positive and negative events do not interfere with each other's amount for each frame.
#     For each frame it also figures out where the center is and returns
#     a fixed sized frame based on the average size of an element.
#     :param positive_event_array:
#     :param negative_event_array:
#     :param total_frames:
#     :param img_shape:
#     :return:
#     """
#     img_height, img_width = img_shape

#     x_data_pos, y_data_pos, z_data_pos, time_data_pos = positive_event_array
#     x_data_neg, y_data_neg, z_data_neg, time_data_neg = negative_event_array

#     window_len_pos = int(len(time_data_pos) / total_frames)
#     window_len_neg = int(len(time_data_neg) / total_frames)

#     frames, len_arr_x, len_arr_y, center_indices, time_frames = [], [], [], [], []
#     i, j = 0, 0

#     while i < len(time_data_pos) and j < len(time_data_neg):
#         current_i, current_j = i, j
#         current_frame = np.zeros((img_height, img_width, 3), np.uint8)
#         time_frame = np.zeros((img_height, img_width, 1), np.uint8)
#         # current_frame.fill(255)
#         while i < len(time_data_pos) and i < current_i + window_len_pos:
#             x = x_data_pos[i]
#             y = y_data_pos[i]
#             current_frame[y][x] = (255, 0, 0)   # Blue
#             time_frame[y][x] = i - current_i    # Latest pixel gets saved
#             i += 1
#         while j < len(time_data_neg) and j < current_j + window_len_neg:
#             x = x_data_neg[j]
#             y = y_data_neg[j]
#             current_frame[y][x] = (0, 0, 255)   # Red
#             time_frame[y][x] = j - current_j    # Latest pixel gets saved
#             j += 1

#         # print('positive time:', i, current_i)
#         # print('negative time:', j, current_j)

#         equalized_hist = cv2.equalizeHist(time_frame)

#         # create a CLAHE object
#         # clahe = cv2.createCLAHE(clipLimit=5.0)
#         # clahe_frame = clahe.apply(time_frame)
#         #
#         # cv2.imshow('time_frame', time_frame)
#         # cv2.imshow('equalized', equalized_hist)
#         # cv2.imshow('clahe_frame', clahe_frame)
#         # cv2.waitKey(200)

#         len_x, len_y, overall_x, overall_y = \
#             gen_extremities(x_data_pos, y_data_pos, x_data_neg, y_data_neg, current_i, i, current_j, j)

#         # Center of the number
#         # current_frame[overall_y][overall_x] = (255, 0, 255)

#         # print(current_frame.shape[1])

#         if np.count_nonzero(current_frame) > 50 \
#                 and current_frame.shape[0] > 0 and current_frame.shape[1] > 0:
#             frames.append(current_frame)
#             len_arr_x.append(len_x)
#             len_arr_y.append(len_y)
#             center_indices.append((overall_x, overall_y))
#             time_frames.append(equalized_hist)

#     cropped_frames, hx, hy, cropping_positions = \
#         generate_cropped_frames(len_arr_x, len_arr_y, frames, center_indices)


#     # print(frame.shape)
#     # print(hx * 2 + 1, hy * 2 + 1, x1 + 1 - x0, y1 + 1 - y0)
#     return frames, cropped_frames, hx * 2 + 1, hy * 2 + 1, cropping_positions, time_frames
