import numpy as np
import statistics
import math
import cv2
# import pdb # For pdb.set_trace() if preferred

# --- generate_event_arrays, gen_extremities, generate_cropped_frames (Keep these as they are) ---
def generate_event_arrays(events, polarity, div_rate=300, mult_rate=100):
    filtered_events = list(filter(lambda entry: entry[3] == polarity, events))
    x_data = [entry[0] for entry in filtered_events]
    y_data = [entry[1] for entry in filtered_events]
    z_data = [int(i / div_rate) * mult_rate for i, entry in enumerate(filtered_events)]
    time_data = [int(entry[2]) for entry in filtered_events] # Keep as microseconds
    return np.array(x_data), np.array(y_data), np.array(z_data), np.array(time_data)


def gen_extremities(x_data_pos, y_data_pos, x_data_neg, y_data_neg,
                    current_i, i, current_j, j):
    all_x_coords = []
    all_y_coords = []

    if i > current_i and len(x_data_pos) > 0 :
        slice_x_pos = x_data_pos[current_i:i]
        slice_y_pos = y_data_pos[current_i:i]
        if len(slice_x_pos) > 0:
            all_x_coords.extend(slice_x_pos)
            all_y_coords.extend(slice_y_pos)
    
    if j > current_j and len(x_data_neg) > 0:
        slice_x_neg = x_data_neg[current_j:j]
        slice_y_neg = y_data_neg[current_j:j]
        if len(slice_x_neg) > 0:
            all_x_coords.extend(slice_x_neg)
            all_y_coords.extend(slice_y_neg)

    if not all_x_coords:
        return 0, 0, 0, 0 

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
    # print(f"DEBUG: generate_cropped_frames: len_arr_x (first 5): {len_arr_x[:5]}, len_arr_y (first 5): {len_arr_y[:5]}")
    # print(f"DEBUG: generate_cropped_frames: Number of frames_list: {len(frames_list)}, center_indices_list (first 5): {center_indices_list[:5]}")
    if not len_arr_x or not len_arr_y:
        # print("DEBUG: generate_cropped_frames: Empty len_arr_x or len_arr_y, returning empty.")
        return [], 0, 0, []

    valid_len_x = [lx for lx in len_arr_x if lx > 0]
    valid_len_y = [ly for ly in len_arr_y if ly > 0]
    # print(f"DEBUG: generate_cropped_frames: valid_len_x count: {len(valid_len_x)}, valid_len_y count: {len(valid_len_y)}")

    if not valid_len_x or not valid_len_y: 
         hx, hy = 10, 10 
         # print(f"DEBUG: generate_cropped_frames: No valid event extents, using default hx={hx}, hy={hy}")
    else:
        mean_len_x = statistics.mean(valid_len_x)
        mean_len_y = statistics.mean(valid_len_y)
        hx, hy = math.ceil(mean_len_x / 2), math.ceil(mean_len_y / 2)
        hx = max(1, hx)
        hy = max(1, hy)
        # print(f"DEBUG: generate_cropped_frames: Calculated mean_len_x={mean_len_x:.2f}, mean_len_y={mean_len_y:.2f}, hx={hx}, hy={hy}")

    cropped_frames_list, cropping_positions_list = [], []
    for ind, frame_item in enumerate(frames_list):
        (cx, cy) = center_indices_list[ind]
        
        y0 = max(0, cy - hy)
        y1 = min(frame_item.shape[0], cy + hy + 1) 
        x0 = max(0, cx - hx)
        x1 = min(frame_item.shape[1], cx + hx + 1)

        if y1 <= y0: y1 = y0 + 1
        if x1 <= x0: x1 = x0 + 1
        
        y1 = min(frame_item.shape[0], y1)
        x1 = min(frame_item.shape[1], x1)

        cropped = frame_item[y0:y1, x0:x1]
        # if ind < 5: # Print for first few frames (LOOP PRINT - REMOVED/COMMENTED)
            # print(f"DEBUG: generate_cropped_frames: Frame {ind}, cx={cx}, cy={cy}, crop_coords: y0={y0},y1={y1},x0={x0},x1={x1}, cropped_shape={cropped.shape}")

        if cropped.size == 0: 
            # print(f"DEBUG: Warning: Empty crop for frame {ind} at cx={cx},cy={cy} with hx={hx},hy={hy}. Orig shape={frame_item.shape}. Using fallback.")
            cropped = frame_item[0:min(10, frame_item.shape[0]), 0:min(10, frame_item.shape[1])] 
            if cropped.size == 0: 
                # print(f"DEBUG: Fallback crop also empty. Using zeros.")
                cropped = np.zeros((1,1,frame_item.shape[2] if len(frame_item.shape) == 3 else 1), dtype=frame_item.dtype)

        cropped_frames_list.append(cropped)
        cropping_positions_list.append((y0, y1, x0, x1)) 

    # print(f"DEBUG: generate_cropped_frames: Returning {len(cropped_frames_list)} cropped_frames. hx={hx}, hy={hy}")
    return cropped_frames_list, hx, hy, cropping_positions_list


# --- Helper function for Time Surface generation (internal to this module) ---
def _create_single_time_surface(event_xs, event_ys, event_ts,
                                t_end_window, tau, img_height, img_width, polarity_str=""):
    # All prints removed from this function as it's called inside the main loop
    surface = np.full((img_height, img_width), -np.inf, dtype=np.float64)
    
    if event_ts.size == 0:
        return np.zeros((img_height, img_width), dtype=np.float32)

    for i in range(event_ts.size):
        x, y, t = int(event_xs[i]), int(event_ys[i]), event_ts[i]
        if 0 <= y < img_height and 0 <= x < img_width:
            surface[y, x] = max(surface[y, x], t)
            
    dt = t_end_window - surface
    exp_surface = np.exp(-dt / tau)
    exp_surface[np.isinf(surface)] = 0 
    
    min_val, max_val = np.min(exp_surface), np.max(exp_surface)
    if max_val > min_val:
        exp_surface = (exp_surface - min_val) / (max_val - min_val)
    else: 
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

    print(f"\n--- generate_event_frames_with_fixed_time_window ---") # SUMMARY PRINT
    print(f"DEBUG: Params: window_len={window_len}us, img_shape={img_shape}, mode='{representation_mode}', tau_on={tau_on}us, tau_off={tau_off}us") # SUMMARY PRINT
    
    img_height, img_width = img_shape

    x_data_pos_den, y_data_pos_den, _, time_data_pos_den = positive_event_array_denoised
    x_data_neg_den, y_data_neg_den, _, time_data_neg_den = negative_event_array_denoised
    
    x_data_pos, y_data_pos, _, time_data_pos = positive_event_array
    x_data_neg, y_data_neg, _, time_data_neg = negative_event_array

    print(f"DEBUG: Denoised Pos Events: count={len(x_data_pos_den)}, time range=[{np.min(time_data_pos_den) if len(time_data_pos_den)>0 else 'N/A'}, {np.max(time_data_pos_den) if len(time_data_pos_den)>0 else 'N/A'}]") # SUMMARY PRINT
    print(f"DEBUG: Denoised Neg Events: count={len(x_data_neg_den)}, time range=[{np.min(time_data_neg_den) if len(time_data_neg_den)>0 else 'N/A'}, {np.max(time_data_neg_den) if len(time_data_neg_den)>0 else 'N/A'}]") # SUMMARY PRINT
    if representation_mode == "rgbd":
        print(f"DEBUG: Noisy Pos Events: count={len(x_data_pos)}, time range=[{np.min(time_data_pos) if len(time_data_pos)>0 else 'N/A'}, {np.max(time_data_pos) if len(time_data_pos)>0 else 'N/A'}]") # SUMMARY PRINT
        print(f"DEBUG: Noisy Neg Events: count={len(x_data_neg)}, time range=[{np.min(time_data_neg) if len(time_data_neg)>0 else 'N/A'}, {np.max(time_data_neg) if len(time_data_neg)>0 else 'N/A'}]") # SUMMARY PRINT

    output_main_frames = []       
    output_denoised_polarity_frames = [] 
    output_depth_compatible_frames = [] 

    len_arr_x_list, len_arr_y_list, center_indices_list = [], [], []
    
    idx_den_pos, idx_den_neg = 0, 0
    idx_noisy_pos, idx_noisy_neg = 0, 0

    all_denoised_times = []
    if time_data_pos_den.size > 0: all_denoised_times.extend(time_data_pos_den)
    if time_data_neg_den.size > 0: all_denoised_times.extend(time_data_neg_den)

    if not all_denoised_times:
        print("DEBUG: No denoised events available to define time windows. Returning empty.") # SUMMARY PRINT
        return [], [], [], 0, 0, [], []

    min_overall_time = np.min(all_denoised_times)
    max_overall_time = np.max(all_denoised_times)
    print(f"DEBUG: Overall denoised time range: [{min_overall_time}, {max_overall_time}]. Duration: {max_overall_time - min_overall_time} us") # SUMMARY PRINT
    
    current_window_start_time = min_overall_time
    window_idx = 0

    # --- START OF MAIN WINDOW LOOP ---
    # All per-window detailed prints have been removed from this loop
    while current_window_start_time <= max_overall_time:
        current_window_end_time = current_window_start_time + window_len
        # print(f"\nDEBUG: [Window {window_idx}] Time: [{current_window_start_time}, {current_window_end_time})") # LOOP PRINT - REMOVED
        # print(f"DEBUG: [Window {window_idx}] Denoised Idx Start: pos={idx_den_pos}, neg={idx_den_neg}") # LOOP PRINT - REMOVED
        # if representation_mode == "rgbd": # LOOP PRINT - REMOVED
            # print(f"DEBUG: [Window {window_idx}] Noisy Idx Start: pos={idx_noisy_pos}, neg={idx_noisy_neg}") # LOOP PRINT - REMOVED

        current_denoised_polarity_frame = np.zeros((img_height, img_width, 3), np.uint8)
        temp_idx_den_pos_start_win = idx_den_pos
        temp_idx_den_neg_start_win = idx_den_neg
        
        end_idx_den_pos_win = idx_den_pos
        while end_idx_den_pos_win < time_data_pos_den.size and \
              time_data_pos_den[end_idx_den_pos_win] < current_window_end_time:
            if time_data_pos_den[end_idx_den_pos_win] >= current_window_start_time:
                x, y = int(x_data_pos_den[end_idx_den_pos_win]), int(y_data_pos_den[end_idx_den_pos_win])
                if 0 <= y < img_height and 0 <= x < img_width:
                    current_denoised_polarity_frame[y, x] = (255, 0, 0)
            end_idx_den_pos_win += 1
        
        end_idx_den_neg_win = idx_den_neg
        while end_idx_den_neg_win < time_data_neg_den.size and \
              time_data_neg_den[end_idx_den_neg_win] < current_window_end_time:
            if time_data_neg_den[end_idx_den_neg_win] >= current_window_start_time:
                x, y = int(x_data_neg_den[end_idx_den_neg_win]), int(y_data_neg_den[end_idx_den_neg_win])
                if 0 <= y < img_height and 0 <= x < img_width:
                    current_denoised_polarity_frame[y, x] = (0, 0, 255)
            end_idx_den_neg_win += 1
        
        den_pos_times_in_potential_range = time_data_pos_den[temp_idx_den_pos_start_win:end_idx_den_pos_win]
        num_den_pos_events_in_win = np.count_nonzero(den_pos_times_in_potential_range >= current_window_start_time)
        den_neg_times_in_potential_range = time_data_neg_den[temp_idx_den_neg_start_win:end_idx_den_neg_win]
        num_den_neg_events_in_win = np.count_nonzero(den_neg_times_in_potential_range >= current_window_start_time)

        # print(f"DEBUG: [Window {window_idx}] Denoised events processed: ...") # LOOP PRINT - REMOVED
        # print(f"DEBUG: [Window {window_idx}] Denoised events IN WINDOW: ...") # LOOP PRINT - REMOVED
        # print(f"DEBUG: [Window {window_idx}] current_denoised_polarity_frame non-zero pixels: ...") # LOOP PRINT - REMOVED

        main_frame_for_this_window = np.zeros((img_height, img_width, 3), np.uint8)
        depth_frame_for_this_window = np.zeros((img_height, img_width, 1), np.uint8)

        if representation_mode == "ts":
            # print(f"DEBUG: [Window {window_idx}] Mode: 'ts'") # LOOP PRINT - REMOVED
            ts_pos_mask = (time_data_pos_den[temp_idx_den_pos_start_win:end_idx_den_pos_win] >= current_window_start_time)
            ts_pos_xs = x_data_pos_den[temp_idx_den_pos_start_win:end_idx_den_pos_win][ts_pos_mask]
            ts_pos_ys = y_data_pos_den[temp_idx_den_pos_start_win:end_idx_den_pos_win][ts_pos_mask]
            ts_pos_ts = time_data_pos_den[temp_idx_den_pos_start_win:end_idx_den_pos_win][ts_pos_mask]

            ts_neg_mask = (time_data_neg_den[temp_idx_den_neg_start_win:end_idx_den_neg_win] >= current_window_start_time)
            ts_neg_xs = x_data_neg_den[temp_idx_den_neg_start_win:end_idx_den_neg_win][ts_neg_mask]
            ts_neg_ys = y_data_neg_den[temp_idx_den_neg_start_win:end_idx_den_neg_win][ts_neg_mask]
            ts_neg_ts = time_data_neg_den[temp_idx_den_neg_start_win:end_idx_den_neg_win][ts_neg_mask]
            # print(f"DEBUG: [Window {window_idx}] TS Input: ...") # LOOP PRINT - REMOVED

            ts_on_surface = _create_single_time_surface(ts_pos_xs, ts_pos_ys, ts_pos_ts,
                                                        current_window_end_time, tau_on, img_height, img_width, polarity_str="ON")
            ts_off_surface = _create_single_time_surface(ts_neg_xs, ts_neg_ys, ts_neg_ts,
                                                         current_window_end_time, tau_off, img_height, img_width, polarity_str="OFF")
            
            main_frame_for_this_window[:, :, 0] = (ts_on_surface * 255).astype(np.uint8)
            main_frame_for_this_window[:, :, 2] = (ts_off_surface * 255).astype(np.uint8)
            # print(f"DEBUG: [Window {window_idx}] TS Output: ...") # LOOP PRINT - REMOVED

        elif representation_mode == "rgbd":
            # print(f"DEBUG: [Window {window_idx}] Mode: 'rgbd'") # LOOP PRINT - REMOVED
            temp_idx_noisy_pos_start_win = idx_noisy_pos
            temp_idx_noisy_neg_start_win = idx_noisy_neg
            noisy_pos_events_in_win_count = 0
            current_noisy_pos_ptr = idx_noisy_pos
            while current_noisy_pos_ptr < time_data_pos.size and time_data_pos[current_noisy_pos_ptr] < current_window_end_time:
                if time_data_pos[current_noisy_pos_ptr] >= current_window_start_time:
                    x, y = int(x_data_pos[current_noisy_pos_ptr]), int(y_data_pos[current_noisy_pos_ptr])
                    if 0 <= y < img_height and 0 <= x < img_width:
                        main_frame_for_this_window[y, x] = (255, 0, 0)
                        depth_frame_for_this_window[y, x, 0] = (current_noisy_pos_ptr - temp_idx_noisy_pos_start_win + 1) 
                    noisy_pos_events_in_win_count +=1
                current_noisy_pos_ptr += 1
            idx_noisy_pos = current_noisy_pos_ptr

            noisy_neg_events_in_win_count = 0
            current_noisy_neg_ptr = idx_noisy_neg
            while current_noisy_neg_ptr < time_data_neg.size and time_data_neg[current_noisy_neg_ptr] < current_window_end_time:
                if time_data_neg[current_noisy_neg_ptr] >= current_window_start_time:
                    x, y = int(x_data_neg[current_noisy_neg_ptr]), int(y_data_neg[current_noisy_neg_ptr])
                    if 0 <= y < img_height and 0 <= x < img_width:
                        main_frame_for_this_window[y, x] = (0, 0, 255)
                        depth_frame_for_this_window[y, x, 0] = max(depth_frame_for_this_window[y,x,0], (current_noisy_neg_ptr - temp_idx_noisy_neg_start_win + 1))
                    noisy_neg_events_in_win_count += 1
                current_noisy_neg_ptr += 1
            idx_noisy_neg = current_noisy_neg_ptr
            
            # print(f"DEBUG: [Window {window_idx}] RGBD: Noisy events processed: ...") # LOOP PRINT - REMOVED
            # print(f"DEBUG: [Window {window_idx}] RGBD Output: main_frame ON non-zero: ...") # LOOP PRINT - REMOVED
            # print(f"DEBUG: [Window {window_idx}] RGBD Depth: depth_frame non-zero before eq: ...") # LOOP PRINT - REMOVED

            if np.count_nonzero(depth_frame_for_this_window) > 0:
                squeezed_depth = depth_frame_for_this_window.squeeze().astype(np.uint8)
                equalized_hist_depth = cv2.equalizeHist(squeezed_depth)
                x_pixels, y_pixels = np.where(equalized_hist_depth > 0)
                scaled_values = equalized_hist_depth[x_pixels, y_pixels] / 2 + 50
                equalized_hist_depth[x_pixels, y_pixels] = np.clip(scaled_values, 0, 255).astype(np.uint8)
                depth_frame_for_this_window = equalized_hist_depth.reshape((img_height, img_width, 1))
                # print(f"DEBUG: [Window {window_idx}] RGBD Depth: depth_frame non-zero after eq: ...") # LOOP PRINT - REMOVED
        else:
            # This error should ideally be caught earlier or handled robustly
            print(f"ERROR: Unknown representation_mode: {representation_mode}") 
            raise ValueError(f"Unknown representation_mode: {representation_mode}")

        processed_any_denoised_in_win = (num_den_pos_events_in_win > 0 or num_den_neg_events_in_win > 0)
        cond1_main_frame_sufficient_events = np.count_nonzero(main_frame_for_this_window) > 100
        cond2_denoised_frame_has_events = np.count_nonzero(current_denoised_polarity_frame) > 0
        cond3_processed_any_denoised = processed_any_denoised_in_win

        # print(f"DEBUG: [Window {window_idx}] Save Condition Check:") # LOOP PRINT - REMOVED
        # print(f"DEBUG:    1. ... -> {cond1_main_frame_sufficient_events}") # LOOP PRINT - REMOVED
        # print(f"DEBUG:    2. ... -> {cond2_denoised_frame_has_events}") # LOOP PRINT - REMOVED
        # print(f"DEBUG:    3. ... -> {cond3_processed_any_denoised}") # LOOP PRINT - REMOVED
        
        if cond1_main_frame_sufficient_events and cond2_denoised_frame_has_events and cond3_processed_any_denoised:
            # print(f"DEBUG: [Window {window_idx}] ALL CONDITIONS MET. Saving frame.") # LOOP PRINT - REMOVED
            len_x, len_y, overall_x, overall_y = gen_extremities(
                x_data_pos_den, y_data_pos_den, x_data_neg_den, y_data_neg_den,
                temp_idx_den_pos_start_win, end_idx_den_pos_win, 
                temp_idx_den_neg_start_win, end_idx_den_neg_win 
            )
            # print(f"DEBUG: [Window {window_idx}] gen_extremities returned: ...") # LOOP PRINT - REMOVED

            output_main_frames.append(main_frame_for_this_window)
            output_denoised_polarity_frames.append(current_denoised_polarity_frame)
            output_depth_compatible_frames.append(depth_frame_for_this_window)
            len_arr_x_list.append(len_x)
            len_arr_y_list.append(len_y)
            center_indices_list.append((overall_x, overall_y))
        # else: # LOOP PRINT - REMOVED
            # print(f"DEBUG: [Window {window_idx}] CONDITIONS NOT MET. Frame not saved.") # LOOP PRINT - REMOVED

        idx_den_pos = end_idx_den_pos_win
        idx_den_neg = end_idx_den_neg_win
        current_window_start_time += window_len 
        window_idx += 1
    # --- END OF MAIN WINDOW LOOP ---


    print(f"\nDEBUG: --- End of Window Loop ---") # SUMMARY PRINT
    print(f"DEBUG: Total windows processed: {window_idx}") # SUMMARY PRINT
    print(f"DEBUG: Number of output_main_frames generated: {len(output_main_frames)}") # SUMMARY PRINT

    if not output_main_frames:
        print("DEBUG: No frames were generated that met the saving criteria. Returning empty lists.") # SUMMARY PRINT
        # PAUSE POINT before returning empty
        input("DEBUG: End of function (no frames generated). Press Enter to exit generate_event_frames_with_fixed_time_window...")
        return [], [], [], 0, 0, [], []

    print(f"DEBUG: Calling generate_cropped_frames with {len(len_arr_x_list)} entries for cropping.") # SUMMARY PRINT
    cropped_output_frames, hx_crop, hy_crop, cropping_positions_list = \
        generate_cropped_frames(len_arr_x_list, len_arr_y_list, output_denoised_polarity_frames, center_indices_list)
    print(f"DEBUG: generate_cropped_frames returned {len(cropped_output_frames)} cropped frames. Crop_half_width (hx_crop)={hx_crop}, Crop_half_height (hy_crop)={hy_crop}") # SUMMARY PRINT

    final_crop_width = hx_crop * 2 + 1 if hx_crop > 0 else 0 
    final_crop_height = hy_crop * 2 + 1 if hy_crop > 0 else 0
    
    print(f"DEBUG: Final returned crop dims: width={final_crop_width}, height={final_crop_height}") # SUMMARY PRINT
    print(f"--- generate_event_frames_with_fixed_time_window finished ---") # SUMMARY PRINT

    # PAUSE POINT before returning results
    input("DEBUG: End of function. Press Enter to exit generate_event_frames_with_fixed_time_window...")

    while True:
        i = 1

    return output_main_frames, output_denoised_polarity_frames, cropped_output_frames, \
           final_crop_width, final_crop_height, cropping_positions_list, output_depth_compatible_frames


# --- generate_fixed_num_events_frames (Keep as is, or adapt similarly if needed) ---
def generate_fixed_num_events_frames(positive_event_array, negative_event_array, total_frames=20, img_shape=(34, 34)):
    img_height, img_width = img_shape
    x_data_pos, y_data_pos, z_data_pos, time_data_pos = positive_event_array
    x_data_neg, y_data_neg, z_data_neg, time_data_neg = negative_event_array

    if len(time_data_pos) == 0 or len(time_data_neg) == 0 or total_frames == 0:
        return [], [], 0, 0, [], [] 
    
    window_len_pos = max(1, int(len(time_data_pos) / total_frames)) 
    window_len_neg = max(1, int(len(time_data_neg) / total_frames))

    frames, len_arr_x, len_arr_y, center_indices, time_frames_list = [], [], [], [], []
    i_ptr, j_ptr = 0, 0 # Renamed i, j to avoid conflict with loop iterators if any

    while i_ptr < len(time_data_pos) and j_ptr < len(time_data_neg):
        current_i, current_j = i_ptr, j_ptr
        current_frame = np.zeros((img_height, img_width, 3), np.uint8)
        time_frame_single = np.zeros((img_height, img_width, 1), np.uint8)

        end_i = min(len(time_data_pos), current_i + window_len_pos)
        for k_pos in range(current_i, end_i):
            x, y = int(x_data_pos[k_pos]), int(y_data_pos[k_pos])
            if 0 <= y < img_height and 0 <= x < img_width:
                current_frame[y, x] = (255, 0, 0)   # Blue
                time_frame_single[y, x, 0] = (k_pos - current_i + 1) 
        i_ptr = end_i
        
        end_j = min(len(time_data_neg), current_j + window_len_neg)
        for k_neg in range(current_j, end_j):
            x, y = int(x_data_neg[k_neg]), int(y_data_neg[k_neg])
            if 0 <= y < img_height and 0 <= x < img_width:
                current_frame[y, x] = (0, 0, 255)   # Red
                time_frame_single[y, x, 0] = max(time_frame_single[y,x,0], (k_neg - current_j + 1)) 
        j_ptr = end_j
        
        if np.count_nonzero(current_frame) > 50:
            squeezed_time_frame = time_frame_single.squeeze().astype(np.uint8)
            if np.any(squeezed_time_frame): 
                 equalized_hist = cv2.equalizeHist(squeezed_time_frame)
            else:
                 equalized_hist = squeezed_time_frame
            
            len_x, len_y, overall_x, overall_y = gen_extremities(
                x_data_pos, y_data_pos, x_data_neg, y_data_neg, 
                current_i, i_ptr, current_j, j_ptr # Use updated i_ptr, j_ptr for end of window
            )
            frames.append(current_frame)
            len_arr_x.append(len_x)
            len_arr_y.append(len_y)
            center_indices.append((overall_x, overall_y))
            time_frames_list.append(equalized_hist.reshape(img_height,img_width,1)) 

    if not frames:
        return [], [], 0, 0, [], []
        
    cropped_frames_list, hx, hy, cropping_positions_list = \
        generate_cropped_frames(len_arr_x, len_arr_y, frames, center_indices)
    
    final_crop_width = hx * 2 + 1 if hx > 0 else 0
    final_crop_height = hy * 2 + 1 if hy > 0 else 0

    return frames, cropped_frames_list, final_crop_width, final_crop_height, cropping_positions_list, time_frames_list





























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
