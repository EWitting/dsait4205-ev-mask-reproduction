import numpy as np
import statistics
import math
import cv2

# # --- Helper for printing array summaries ---
# def print_array_summary(name, arr):
#     if isinstance(arr, (list, tuple)) and len(arr) > 0 and isinstance(arr[0], np.ndarray):
#         # Print summary for list of numpy arrays
#         non_empty_shapes = [item.shape for item in arr if hasattr(item, 'shape')]
#         if non_empty_shapes:
#             print(f"DEBUG: {name}: list of {len(arr)} np.arrays, first item shape: {arr[0].shape if len(arr) > 0 else 'N/A'}, typical shape: {statistics.mode(non_empty_shapes) if non_empty_shapes else 'N/A'}")
#         else:
#             print(f"DEBUG: {name}: list of {len(arr)} items (likely empty or non-array items)")
#     elif isinstance(arr, np.ndarray):
#         print(f"DEBUG: {name}: shape: {arr.shape}, dtype: {arr.dtype}, min: {np.min(arr) if arr.size > 0 else 'N/A'}, max: {np.max(arr) if arr.size > 0 else 'N/A'}")
#     elif isinstance(arr, (list, tuple)):
#         print(f"DEBUG: {name}: length: {len(arr)}")
#         if 0 < len(arr) < 6: # Print a few elements if the list is short and not empty
#             print(f"DEBUG: {name} (first few elements): {arr[:5]}")
#     else:
#         print(f"DEBUG: {name}: {arr}")

def generate_event_arrays(events, polarity, div_rate=300, mult_rate=100):
    """
    Splits the array of events into separate arrays for x, y, time, and a custom z-coordinate,
    based on the specified polarity.

    Input:
    events = stream of events, one entry looks like: (x, y, time, polarity)
    polarity = 1 for positive events, 0 for negative events
    div_rate = Divisor for calculating z_data from event index. Used for flattening/grouping events.
               If div_rate=1, mult_rate=1, z_data is the event index.
    mult_rate = Multiplier for calculating z_data from event index. Scales z_data.

    Output (same indices for same event within the filtered set):
    x_data = array with x-coordinates of the filtered events
    y_data = array with y-coordinates of the filtered events
    z_data = array with custom z-coordinates. Derived from the *index* of the event in the
             filtered list (not its timestamp). Can represent "event order depth" or grouping.
    time_data = array with timestamps of the filtered events, converted from microseconds to milliseconds.
    """
    # print(f"\nDEBUG: --- Entering generate_event_arrays ---")
    # print(f"DEBUG: Input total events: {len(events)}, target polarity: {polarity}")
    # print(f"DEBUG: div_rate: {div_rate}, mult_rate: {mult_rate}")

    # Filter events based on the given polarity
    filtered_events = list(filter(lambda entry: entry[3] == polarity, events))
    # print(f"DEBUG: Number of events after filtering for polarity {polarity}: {len(filtered_events)}")

    if not filtered_events:
        # print(f"DEBUG: No events found for polarity {polarity}. Returning empty lists.")
        # print(f"DEBUG: --- Exiting generate_event_arrays ---")
        return [], [], [], []

    # Extract x and y coordinates
    x_data = [entry[0] for entry in filtered_events]
    y_data = [entry[1] for entry in filtered_events]

    # Calculate z_data based on the index of the event in the filtered list.
    # This creates a form of "event packet index" or "temporal grouping" based on event count.
    # e.g., if div_rate=300, mult_rate=100:
    #   Events 0-299 (filtered index) -> z = int(index/300)*100 = 0
    #   Events 300-599 (filtered index) -> z = int(index/300)*100 = 100
    z_data = [int(i / div_rate) * mult_rate for i, entry in enumerate(filtered_events)]

    # Convert timestamps from microseconds to milliseconds
    time_data = [int(entry[2] / 1000) for entry in filtered_events]

    # print_array_summary("x_data", x_data)
    # print_array_summary("y_data", y_data)
    # print_array_summary("z_data (custom event order based)", z_data)
    # if z_data: print(f"DEBUG: Sample z_data (first 5 if available): {z_data[:5]}")
    # print_array_summary("time_data (timestamps in ms)", time_data)
    # if time_data: print(f"DEBUG: Sample time_data (first 5 if available, in ms): {time_data[:5]}")
    # print(f"DEBUG: --- Exiting generate_event_arrays ---")

    return x_data, y_data, z_data, time_data


# def generate_event_frames(positive_event_array, negative_event_array, window_len=10, img_shape=(34, 34)):
#     """
#     Generates frames by accumulating events within fixed time windows.
#     Positive events are colored blue, negative events are red.

#     :param positive_event_array: Tuple of (x_data_pos, y_data_pos, z_data_pos, time_data_pos) for positive events.
#     :param negative_event_array: Tuple of (x_data_neg, y_data_neg, z_data_neg, time_data_neg) for negative events.
#     :param window_len: Duration (in milliseconds) of the time window to accumulate events for each frame.
#     :param img_shape: Tuple (height, width) specifying the dimensions of the output frames.
#     :return: A list of generated frames (numpy arrays, BGR format).
#     """
#     print(f"\nDEBUG: --- Entering generate_event_frames ---")
#     print(f"DEBUG: window_len (ms): {window_len}, img_shape: {img_shape}")

#     img_height, img_width = img_shape
#     frames = []
#     i, j = 0, 0  # Pointers for positive and negative event arrays respectively

#     # Unpack event data. z_data is not used in this specific frame generation function.
#     x_data_pos, y_data_pos, _, time_data_pos = positive_event_array
#     x_data_neg, y_data_neg, _, time_data_neg = negative_event_array

#     print_array_summary("time_data_pos (ms)", time_data_pos)
#     print_array_summary("time_data_neg (ms)", time_data_neg)

#     if not time_data_pos or not time_data_neg:
#         print("DEBUG: One or both event streams are empty. No frames will be generated.")
#         print(f"DEBUG: --- Exiting generate_event_frames ---")
#         return []

#     first_frame_logged = False
#     # Loop as long as there are events in both positive and negative streams to process
#     while i < len(time_data_pos) and j < len(time_data_neg):
#         # Determine the start time for the current frame's window.
#         # This is the timestamp of the earlier event between the current positive and negative event.
#         current_time = min(time_data_pos[i], time_data_neg[j])
        
#         # Initialize a new blank frame (black background, 3 channels for BGR color)
#         current_frame = np.zeros((img_height, img_width, 3), np.uint8)
#         # To use a white background instead: current_frame.fill(255)

#         start_i_for_window = i
#         start_j_for_window = j

#         # Accumulate positive events that fall within the current time window
#         # The window extends from `current_time` to `current_time + window_len - 1` (inclusive).
#         while i < len(time_data_pos) and time_data_pos[i] < current_time + window_len:
#             x = x_data_pos[i]
#             y = y_data_pos[i]
#             # Ensure coordinates are within image bounds before drawing
#             if 0 <= y < img_height and 0 <= x < img_width:
#                  current_frame[y][x] = (255, 0, 0)   # Blue (BGR format) for positive events
#             i += 1
        
#         # Accumulate negative events that fall within the current time window
#         while j < len(time_data_neg) and time_data_neg[j] < current_time + window_len:
#             x = x_data_neg[j]
#             y = y_data_neg[j]
#             if 0 <= y < img_height and 0 <= x < img_width:
#                 current_frame[y][x] = (0, 0, 255)   # Red (BGR format) for negative events
#             j += 1
        
#         # Add the completed frame to the list of frames.
#         # Consider adding a threshold: if np.count_nonzero(current_frame) > min_event_threshold:
#         frames.append(current_frame)
        
#         if not first_frame_logged and frames:
#             print(f"DEBUG: First frame generated for time window starting at {current_time} ms, ending before {current_time + window_len} ms.")
#             print(f"DEBUG:   Positive events in this window: {i - start_i_for_window}")
#             print(f"DEBUG:   Negative events in this window: {j - start_j_for_window}")
#             print_array_summary("  First generated frame", current_frame)
#             first_frame_logged = True

#     # --- Loop Summary ---
#     print(f"DEBUG: generate_event_frames main loop finished.")
#     print(f"DEBUG: Total positive events iterated up to index: {i} (out of {len(time_data_pos)})")
#     print(f"DEBUG: Total negative events iterated up to index: {j} (out of {len(time_data_neg)})")
#     print(f"DEBUG: Total frames generated: {len(frames)}")
#     if frames: print_array_summary("List of generated frames", frames)
    
#     print(f"DEBUG: --- Exiting generate_event_frames ---")
#     return frames


def gen_extremities(x_data_pos, y_data_pos, x_data_neg, y_data_neg,
                    current_i, i_exclusive, current_j, j_exclusive):
    """
    Calculates the bounding box (min/max extremities) and center of events within specified slices
    of positive and negative event data.
    NOTE: The original uncommented code primarily uses NEGATIVE events for these calculations if their slice is non-empty.
          Positive event calculations were commented out. This version includes a basic fallback to positive events.

    :param x_data_pos: List of x-coordinates for positive events.
    :param y_data_pos: List of y-coordinates for positive events.
    :param x_data_neg: List of x-coordinates for negative events.
    :param y_data_neg: List of y-coordinates for negative events.
    :param current_i: Start index (inclusive) for the slice of positive events.
    :param i_exclusive: End index (exclusive) for the slice of positive events.
    :param current_j: Start index (inclusive) for the slice of negative events.
    :param j_exclusive: End index (exclusive) for the slice of negative events.
    :return: Tuple (len_x, len_y, overall_x, overall_y)
             len_x: Width of the bounding box.
             len_y: Height of the bounding box.
             overall_x: Center x-coordinate of the bounding box.
             overall_y: Center y-coordinate of the bounding box.
    """
    # print(f"DEBUG: --- Entering gen_extremities ---")
    # print(f"DEBUG: Positive event slice indices: {current_i} to {i_exclusive-1}")
    # print(f"DEBUG: Negative event slice indices: {current_j} to {j_exclusive-1}")

    # Extract the relevant slices of event data
    pos_x_slice = x_data_pos[current_i:i_exclusive]
    pos_y_slice = y_data_pos[current_i:i_exclusive]
    neg_x_slice = x_data_neg[current_j:j_exclusive]
    neg_y_slice = y_data_neg[current_j:j_exclusive]

    # Initialize default values for extremities and center
    len_x, len_y = 0, 0
    overall_x, overall_y = 0, 0 # Default center, e.g., (0,0) or image center if preferred

    # --- Positive event calculations (original code had this section commented out) ---
    # This section could be used if positive events should also define the bounding box.
    # if pos_x_slice:
    #     pos_mn_x = min(pos_x_slice)
    #     pos_mn_y = min(pos_y_slice)
    #     pos_mx_x = max(pos_x_slice)
    #     pos_mx_y = max(pos_y_slice)
    #     # Center of positive events' bounding box
    #     pos_overall_x = ((pos_mx_x - pos_mn_x) / 2) + pos_mn_x
    #     pos_overall_y = ((pos_mx_y - pos_mn_y) / 2) + pos_mn_y
    # else:
    #     # Handle empty positive slice if combined logic were used
    #     pos_mn_x, pos_mx_x, pos_mn_y, pos_mx_y = float('inf'), float('-inf'), float('inf'), float('-inf')
    #     pos_overall_x, pos_overall_y = 0, 0


    # --- Negative event calculations (primary focus in original code) ---
    if neg_x_slice: # Check if there are any negative events in the slice
        neg_mn_x = min(neg_x_slice)
        neg_mn_y = min(neg_y_slice)
        neg_mx_x = max(neg_x_slice)
        neg_mx_y = max(neg_y_slice)

        # Calculate width (len_x) and height (len_y) of the negative events' bounding box
        len_x = neg_mx_x - neg_mn_x
        len_y = neg_mx_y - neg_mn_y

        # Calculate center of the negative events' bounding box
        neg_overall_x = ((neg_mx_x - neg_mn_x) / 2) + neg_mn_x
        neg_overall_y = ((neg_mx_y - neg_mn_y) / 2) + neg_mn_y
        
        # The original code sets the overall values based purely on negative events if available
        overall_x = int(neg_overall_x)
        overall_y = int(neg_overall_y)
        # print(f"DEBUG: gen_extremities: Using negative events. Min/Max X: ({neg_mn_x},{neg_mx_x}), Y: ({neg_mn_y},{neg_mx_y})")
        # print(f"DEBUG: gen_extremities: Calculated len_x={len_x}, len_y={len_y}, center=({overall_x},{overall_y})")

    elif pos_x_slice: # Fallback: if no negative events, use positive events (not in original's active code)
        # print(f"DEBUG: gen_extremities: No negative events in slice, using positive events for extremities.")
        pos_mn_x = min(pos_x_slice)
        pos_mn_y = min(pos_y_slice)
        pos_mx_x = max(pos_x_slice)
        pos_mx_y = max(pos_y_slice)
        len_x = pos_mx_x - pos_mn_x
        len_y = pos_mx_y - pos_mn_y
        overall_x = int(((pos_mx_x - pos_mn_x) / 2) + pos_mn_x)
        overall_y = int(((pos_mx_y - pos_mn_y) / 2) + pos_mn_y)
        # print(f"DEBUG: gen_extremities: Using positive events. Min/Max X: ({pos_mn_x},{pos_mx_x}), Y: ({pos_mn_y},{pos_mx_y})")
        # print(f"DEBUG: gen_extremities: Calculated len_x={len_x}, len_y={len_y}, center=({overall_x},{overall_y})")
    else:
        print(f"DEBUG: gen_extremities: No events in either positive or negative slices. Extremities remain (0,0,0,0).")
        # len_x, len_y, overall_x, overall_y will remain 0,0,0,0 as initialized

    # --- Original combined logic (was commented out) ---
    # This would be used if you want to find a bounding box encompassing both positive and negative events.
    # if pos_x_slice and neg_x_slice:
    #     # Combined length considering both positive and negative events
    #     # len_x = max(pos_mx_x, neg_mx_x) - min(pos_mn_x, neg_mn_x)
    #     # len_y = max(pos_mx_y, neg_mx_y) - min(pos_mn_y, neg_mn_y)
    #     # Combined center (average of positive and negative centers)
    #     # overall_x = int((pos_overall_x + neg_overall_x) / 2)
    #     # overall_y = int((pos_overall_y + neg_overall_y) / 2)
    # elif pos_x_slice: # Only positive events
    #     # (already handled by fallback logic above if neg_x_slice is empty)
    # elif neg_x_slice: # Only negative events
    #     # (already handled by primary logic above)
    
    # print(f"DEBUG: --- Exiting gen_extremities ---")
    return len_x, len_y, overall_x, overall_y


def generate_cropped_frames(len_arr_x, len_arr_y, frames, center_indices):
    """
    Crops a list of frames to a consistent size, centered around provided coordinates.
    The consistent size is determined by the average width (mean_len_x) and height (mean_len_y)
    of the content in the original frames, derived from len_arr_x and len_arr_y.

    :param len_arr_x: List of content widths for each original frame.
    :param len_arr_y: List of content heights for each original frame.
    :param frames: List of original frames (numpy arrays) to be cropped.
    :param center_indices: List of (cx, cy) tuples, where cx, cy is the center for cropping each frame.
    :return: Tuple (cropped_frames, hx, hy, cropping_positions)
             cropped_frames: List of cropped frames.
             hx: Calculated half-width for cropping (ceil(mean_len_x / 2)).
             hy: Calculated half-height for cropping (ceil(mean_len_y / 2)).
             cropping_positions: List of (y_start_slice, y_end_slice, x_start_slice, x_end_slice) tuples for each crop.
    """
    # print(f"\nDEBUG: --- Entering generate_cropped_frames ---")
    # print_array_summary("Input len_arr_x (content widths)", len_arr_x)
    # print_array_summary("Input len_arr_y (content heights)", len_arr_y)
    # print_array_summary("Input frames for cropping", frames)
    # print_array_summary("Input center_indices for cropping", center_indices)

    # Calculate the average width and height of the content across all frames
    if len(len_arr_x) > 0 and len(len_arr_y) > 0:
        mean_len_x = statistics.mean(len_arr_x)
        mean_len_y = statistics.mean(len_arr_y)
        # Calculate half-width (hx) and half-height (hy) for the target crop box.
        # math.ceil ensures the crop window is large enough to contain the average content.
        hx = math.ceil(mean_len_x / 2)
        hy = math.ceil(mean_len_y / 2)
        # print(f"DEBUG: Calculated mean_len_x: {mean_len_x:.2f}, mean_len_y: {mean_len_y:.2f}")
        # print(f"DEBUG: Calculated crop half-width (hx): {hx}, half-height (hy): {hy}")
    else:
        # print(f"DEBUG: len_arr_x or len_arr_y is empty. Setting hx, hy to 0 (will result in small/empty crops).")
        hx, hy = 0, 0

    cropped_frames, cropping_positions = [], []
    
    if not frames:
        # print("DEBUG: No frames provided to crop.")
        # print(f"DEBUG: --- Exiting generate_cropped_frames ---")
        return [], hx, hy, []

    # print(f"DEBUG: Cropping {len(frames)} frames using hx={hx}, hy={hy}...")
    first_crop_logged = False
    for ind, frame in enumerate(frames):
        if ind >= len(center_indices):
            # print(f"DEBUG: Warning: Not enough center_indices for frame index {ind}. Skipping this frame.")
            continue
        
        (cx, cy) = center_indices[ind] # Center (x,y) for the current frame's content

        # Define the desired crop region based on center (cx,cy) and half-sizes (hx,hy)
        # Desired start coordinates (inclusive)
        desired_x_start = cx - hx
        desired_y_start = cy - hy
        # Desired end coordinates (inclusive)
        desired_x_end_inclusive = cx + hx
        desired_y_end_inclusive = cy + hy

        # Actual slice indices, ensuring they are within the frame's bounds.
        # Slicing uses [start_inclusive : end_exclusive].
        # `final_x0`, `final_y0` are the inclusive start indices for the slice.
        final_x0 = max(0, desired_x_start)
        final_y0 = max(0, desired_y_start)
        
        # `final_x1_exclusive`, `final_y1_exclusive` are the exclusive end indices for the slice.
        # We add 1 to the inclusive end coordinate to make it exclusive for slicing.
        final_x1_exclusive = min(frame.shape[1], desired_x_end_inclusive + 1)
        final_y1_exclusive = min(frame.shape[0], desired_y_end_inclusive + 1)
        
        # Perform the crop. Ensure start index is not greater than end index.
        if final_y0 < final_y1_exclusive and final_x0 < final_x1_exclusive:
            cropped = frame[final_y0 : final_y1_exclusive, final_x0 : final_x1_exclusive]
        else:
            # If crop dimensions are invalid (e.g. hx/hy are too large, or center is off), create an empty array
            # with expected channels, or handle as an error.
            # print(f"DEBUG: Invalid crop dimensions for frame {ind}. y_slice=({final_y0}:{final_y1_exclusive}), x_slice=({final_x0}:{final_x1_exclusive}). Creating empty-like crop.")
            cropped = np.zeros((0, 0, frame.shape[2]) if len(frame.shape) == 3 else (0,0) , dtype=frame.dtype)


        if not first_crop_logged:
            # print(f"DEBUG: First frame crop details (frame index {ind}):")
            # print(f"DEBUG:   Original frame shape: {frame.shape}")
            # print(f"DEBUG:   Center (cx, cy): ({cx}, {cy})")
            # print(f"DEBUG:   Target half-width/height (hx, hy): ({hx}, {hy})")
            # print(f"DEBUG:   Desired crop box (inclusive coords): x=[{desired_x_start}-{desired_x_end_inclusive}], y=[{desired_y_start}-{desired_y_end_inclusive}]")
            # print(f"DEBUG:   Actual slice indices (y_start:y_end_excl, x_start:x_end_excl): y=[{final_y0}:{final_y1_exclusive}], x=[{final_x0}:{final_x1_exclusive}]")
            # print_array_summary("  First cropped frame", cropped)
            first_crop_logged = True

        cropped_frames.append(cropped)
        # Store the actual slice boundaries used (y_start, y_end_exclusive, x_start, x_end_exclusive)
        cropping_positions.append((final_y0, final_y1_exclusive, final_x0, final_x1_exclusive))

    # print_array_summary("List of all cropped frames", cropped_frames)
    # print_array_summary("List of all cropping positions", cropping_positions)
    # print(f"DEBUG: --- Exiting generate_cropped_frames ---")
    return cropped_frames, hx, hy, cropping_positions


def generate_event_frames_with_fixed_time_window(positive_event_array_denoised, negative_event_array_denoised,
                                                 positive_event_array, negative_event_array,
                                                 window_len=20, img_shape=(34, 34), **kwargs):
    """
    Generates frames using a fixed time window.
    Temporal information (time_frames) can be generated in two ways:
    1. 'original': Based on event order, histogram equalization, and offset.
    2. 'actual_time_normalization': Based on actual time since window start, normalized to 0-255.
       Controlled by `representation_mode` in kwargs.

    :param positive_event_array_denoised: Denoised positive events (x, y, z, time_ms).
    :param negative_event_array_denoised: Denoised negative events (x, y, z, time_ms).
    :param positive_event_array: Original (noised) positive events (x, y, z, time_ms).
    :param negative_event_array: Original (noised) negative events (x, y, z, time_ms).
    :param window_len: Duration (in ms) of the time window for event accumulation.
    :param img_shape: Tuple (height, width) for output frames.
    :param kwargs: Additional arguments. Expected: `representation_mode` (str, optional, defaults to 'original').
                   Set to 'actual_time_normalization' for the new method.
    :return: Tuple: frames, frames_denoised, cropped_frames, crop_width, crop_height, cropping_positions, time_frames.
    """
    representation_mode = kwargs.get('representation_mode', 'rgbd_original')
    print(f"DEBUG: Representation mode: {representation_mode}")
    # print(f"\nDEBUG: --- Entering generate_event_frames_with_fixed_time_window ---")
    # print(f"DEBUG: window_len (ms): {window_len}, img_shape: {img_shape}")
    # print(f"DEBUG: Using representation_mode for temporal frames: {representation_mode}")


    img_height, img_width = img_shape

    x_data_pos_den, y_data_pos_den, _, time_data_pos_den = positive_event_array_denoised
    x_data_neg_den, y_data_neg_den, _, time_data_neg_den = negative_event_array_denoised
    x_data_pos, y_data_pos, _, time_data_pos = positive_event_array
    x_data_neg, y_data_neg, _, time_data_neg = negative_event_array
    
    # Initial DEBUG prints for event streams (can be verbose, enable if needed)
    # print("DEBUG: Input Denoised Positive Events:")
    # print_array_summary("  time_data_pos_den", time_data_pos_den)
    # print("DEBUG: Input Denoised Negative Events:")
    # print_array_summary("  time_data_neg_den", time_data_neg_den)
    # print("DEBUG: Input Original (Noised) Positive Events:")
    # print_array_summary("  time_data_pos", time_data_pos)
    # print("DEBUG: Input Original (Noised) Negative Events:")
    # print_array_summary("  time_data_neg", time_data_neg)

    if not time_data_pos_den or not time_data_neg_den:
        # print("DEBUG: Denoised event streams are empty. Cannot reliably determine window start times or extremities. Exiting.")
        # print(f"DEBUG: --- Exiting generate_event_frames_with_fixed_time_window ---")
        return [], [], [], 0, 0, [], []

    frames, len_arr_x, len_arr_y, center_indices, time_frames, frames_denoised = [], [], [], [], [], []
    i_den, j_den = 0, 0
    i, j = 0, 0

    first_valid_frame_logged = False

    while i_den < len(time_data_pos_den) and j_den < len(time_data_neg_den):
        current_i_den, current_j_den = i_den, j_den
        current_i, current_j = i, j
        
        current_frame_noised = np.zeros((img_height, img_width, 3), np.uint8)
        current_frame_denoised_rgb = np.zeros((img_height, img_width, 3), np.uint8)

        # Initialize raw temporal data storage based on representation_mode
        if representation_mode == 'rgbd_depth_time_normalized':
            # Stores actual time deltas from window start [0, window_len-1]
            # Initialize with -1.0 to mark pixels with no events
            raw_temporal_data = np.full((img_height, img_width), -1.0, dtype=np.float32)
        elif representation_mode == 'rgbd_original': # original
            # Stores event order within the window, modulo 255
            raw_temporal_data = np.zeros((img_height, img_width), np.uint8)

        current_time = min(time_data_pos_den[i_den], time_data_neg_den[j_den])
        
        num_pos_den_in_window, num_neg_den_in_window = 0, 0
        while i_den < len(time_data_pos_den) and time_data_pos_den[i_den] < current_time + window_len:
            x, y = x_data_pos_den[i_den], y_data_pos_den[i_den]
            if 0 <= y < img_height and 0 <= x < img_width:
                current_frame_denoised_rgb[y][x] = (255, 0, 0)
            i_den += 1
            num_pos_den_in_window +=1
        
        while j_den < len(time_data_neg_den) and time_data_neg_den[j_den] < current_time + window_len:
            x, y = x_data_neg_den[j_den], y_data_neg_den[j_den]
            if 0 <= y < img_height and 0 <= x < img_width:
                current_frame_denoised_rgb[y][x] = (0, 0, 255)
            j_den += 1
            num_neg_den_in_window += 1

        num_pos_orig_in_window, num_neg_orig_in_window = 0, 0
        # Process Original Positive Events
        while i < len(time_data_pos) and time_data_pos[i] < current_time + window_len:
            x, y = x_data_pos[i], y_data_pos[i]
            event_actual_time_ms = time_data_pos[i]
            if 0 <= y < img_height and 0 <= x < img_width:
                current_frame_noised[y][x] = (255, 0, 0)
                if representation_mode == 'rgbd_depth_time_normalized':
                    delta_t = float(event_actual_time_ms - current_time)
                    raw_temporal_data[y, x] = max(raw_temporal_data[y, x], delta_t)
                elif representation_mode == 'rgbd_original': # original
                    raw_temporal_data[y, x] = max(raw_temporal_data[y, x], ((i - current_i + 1) % 255))
            i += 1
            num_pos_orig_in_window += 1
        
        # Process Original Negative Events
        while j < len(time_data_neg) and time_data_neg[j] < current_time + window_len:
            x, y = x_data_neg[j], y_data_neg[j]
            event_actual_time_ms = time_data_neg[j]
            if 0 <= y < img_height and 0 <= x < img_width:
                current_frame_noised[y][x] = (0, 0, 255)
                if representation_mode == 'rgbd_depth_time_normalized':
                    delta_t = float(event_actual_time_ms - current_time)
                    raw_temporal_data[y, x] = max(raw_temporal_data[y, x], delta_t)
                elif representation_mode == 'rgbd_original': # original
                    raw_temporal_data[y, x] = max(raw_temporal_data[y, x], ((j - current_j + 1) % 255))
            j += 1
            num_neg_orig_in_window += 1

        # --- Post-process raw_temporal_data to create the final_time_frame ---
        if representation_mode == 'rgbd_depth_time_normalized':
            processed_temporal_frame = np.zeros((img_height, img_width), dtype=np.uint8)
            # Pixels with events have raw_temporal_data >= 0
            active_mask = raw_temporal_data >= 0.0

            if np.any(active_mask):
                active_values = raw_temporal_data[active_mask] # Values in [0, window_len-1]

                if window_len > 1:
                    # Normalize from [0, window_len-1] to [0, 255]
                    # Smallest delta_t is 0 (maps to 0), largest is window_len-1 (maps to 255)
                    normalized_active_values = (active_values / (window_len - 1.0)) * 255.0
                elif window_len == 1:
                    # All active events occurred at delta_t = 0. Map to 0 (earliest).
                    normalized_active_values = np.zeros_like(active_values, dtype=np.float32)
                else: # window_len <= 0. Unlikely if events exist for this window. Fallback to 0.
                    normalized_active_values = np.zeros_like(active_values, dtype=np.float32)
                
                processed_temporal_frame[active_mask] = np.clip(normalized_active_values, 0, 255).astype(np.uint8)
            
            final_time_frame = processed_temporal_frame.reshape((img_height, img_width, 1))
        elif representation_mode == 'rgbd_original': # 'original' mode post-processing
            # Ensure raw_temporal_data is uint8 for equalizeHist
            event_order_raw_uint8 = raw_temporal_data.astype(np.uint8)
            if np.any(event_order_raw_uint8 > 0):
                equalized_hist_2d = cv2.equalizeHist(event_order_raw_uint8)
            else:
                equalized_hist_2d = event_order_raw_uint8

            # Apply the /2 + 50 adjustment
            temp_processed_frame = equalized_hist_2d.copy() # Work on a copy
            active_pixels_mask = temp_processed_frame > 0
            # Ensure uint8 for arithmetic robustness
            modified_values = (temp_processed_frame[active_pixels_mask].astype(np.float32) / 2.0).astype(np.uint8) + 50
            temp_processed_frame[active_pixels_mask] = np.clip(modified_values, 0, 255)
            
            final_time_frame = temp_processed_frame.reshape((img_height, img_width, 1))

        # --- Filter and Store Valid Frames ---
        if (np.count_nonzero(current_frame_noised) > 100 and
            np.count_nonzero(current_frame_denoised_rgb) > 0 and
            (i_den > current_i_den and j_den > current_j_den)): 
            
            len_x, len_y, overall_x, overall_y = \
                gen_extremities(x_data_pos_den, y_data_pos_den, x_data_neg_den, y_data_neg_den,
                                current_i_den, i_den, current_j_den, j_den)
            
            frames.append(current_frame_noised)
            frames_denoised.append(current_frame_denoised_rgb)
            len_arr_x.append(len_x)
            len_arr_y.append(len_y)
            center_indices.append((overall_x, overall_y))
            time_frames.append(final_time_frame)

            if not first_valid_frame_logged:
                # print(f"DEBUG: First valid frame generated (fixed time window) at current_time_ms={current_time}")
                # print(f"DEBUG:   Denoised events in window: +{num_pos_den_in_window}, -{num_neg_den_in_window}")
                # print(f"DEBUG:   Original events in window: +{num_pos_orig_in_window}, -{num_neg_orig_in_window}")
                # print_array_summary("  final_time_frame (temporal map)", final_time_frame)
                # print(f"DEBUG:   Calculated extremities from denoised: len_x={len_x}, len_y={len_y}, center=({overall_x},{overall_y})")
                # if raw_temporal_data.size > 0:
                    #  print_array_summary(f"  Raw temporal data (mode: {representation_mode})", raw_temporal_data)
                first_valid_frame_logged = True
        
    # --- Loop Summary ---
    # print(f"DEBUG: generate_event_frames_with_fixed_time_window main loop finished.")
    # print(f"DEBUG: Number of valid frames generated: {len(frames)}")

    if frames_denoised:
        cropped_frames, hx, hy, cropping_positions = \
            generate_cropped_frames(len_arr_x, len_arr_y, frames_denoised, center_indices)
        crop_width = hx * 2 + 1 if hx > 0 else (1 if len(len_arr_x)>0 and statistics.mean(len_arr_x) == 0 else 0) 
        crop_height = hy * 2 + 1 if hy > 0 else (1 if len(len_arr_y)>0 and statistics.mean(len_arr_y) == 0 else 0)
        # print(f"DEBUG: Cropping of denoised frames complete. Target crop_width={crop_width}, crop_height={crop_height}")
    else:
        cropped_frames, cropping_positions = [], []
        crop_width, crop_height = 0,0
        # print(f"DEBUG: No denoised frames generated, no cropping performed.")

    # print(f"DEBUG: --- Exiting generate_event_frames_with_fixed_time_window ---")
    return frames, frames_denoised, cropped_frames, crop_width, crop_height, cropping_positions, time_frames


def generate_fixed_num_events_frames(positive_event_array, negative_event_array, total_frames=20, img_shape=(34, 34)):
    """
    Generates a fixed number of event frames. Each frame accumulates a fixed *number* of positive
    and negative events (a "packet" of events), rather than events within a fixed time window.
    The number of positive and negative events per frame are determined independently based on `total_frames`.
    It also calculates content extremities and center for each frame, and returns cropped versions
    of these frames, normalized to an average content size.
    A 'time_frame' (single channel, pseudo-depth) is also generated, representing event order/recency 
    within each event packet, visualized after histogram equalization.

    :param positive_event_array: Tuple of (x_data_pos, y_data_pos, z_data_pos, time_data_pos).
    :param negative_event_array: Tuple of (x_data_neg, y_data_neg, z_data_neg, time_data_neg).
    :param total_frames: The desired total number of frames to generate.
    :param img_shape: Tuple (height, width) for output frames.
    :return: Tuple:
             frames (list): RGB frames, each with a fixed number of positive/negative events.
             cropped_frames (list): Above frames cropped to a consistent size.
             crop_width (int): Width of the cropped region (typically hx * 2 + 1).
             crop_height (int): Height of the cropped region (typically hy * 2 + 1).
             cropping_positions (list): Slice boundaries for each cropped frame.
             time_frames (list): Grayscale frames representing event order/recency within each packet.
    """
    # print(f"\nDEBUG: --- Entering generate_fixed_num_events_frames ---")
    # print(f"DEBUG: Desired total_frames: {total_frames}, img_shape: {img_shape}")

    img_height, img_width = img_shape

    # Unpack event data. z_data is not used in this frame generation. time_data_pos/neg are used for iteration but not for windowing.
    x_data_pos, y_data_pos, _, time_data_pos = positive_event_array
    x_data_neg, y_data_neg, _, time_data_neg = negative_event_array

    # print("DEBUG: Input Positive Events:")
    # print_array_summary("  time_data_pos", time_data_pos) # Length is important here
    # print("DEBUG: Input Negative Events:")
    # print_array_summary("  time_data_neg", time_data_neg) # Length is important here

    if not time_data_pos or not time_data_neg:
        # print("DEBUG: One or both event streams are empty. Cannot generate frames based on fixed number of events. Exiting.")
        # print(f"DEBUG: --- Exiting generate_fixed_num_events_frames ---")
        return [], [], 0, 0, [], []

    # Calculate number of positive events per frame ("packet size")
    if total_frames > 0 and len(time_data_pos) > 0:
        window_len_pos = max(1, int(len(time_data_pos) / total_frames)) # Ensure at least 1
    elif len(time_data_pos) > 0: # if total_frames is 0 or less, use all events for one frame
        window_len_pos = len(time_data_pos)
    else: # no positive events
        window_len_pos = 0
        
    # Calculate number of negative events per frame ("packet size")
    if total_frames > 0 and len(time_data_neg) > 0:
        window_len_neg = max(1, int(len(time_data_neg) / total_frames)) # Ensure at least 1
    elif len(time_data_neg) > 0: # if total_frames is 0 or less, use all events for one frame
        window_len_neg = len(time_data_neg)
    else: # no negative events
        window_len_neg = 0

    # print(f"DEBUG: Calculated events per frame packet: Positive (window_len_pos)={window_len_pos}, Negative (window_len_neg)={window_len_neg}")
    
    if window_len_pos == 0 or window_len_neg == 0 :
        # print(f"DEBUG: Warning: Calculated event packet size is 0 for positive or negative events. No frames will be generated.")
        # print(f"DEBUG: --- Exiting generate_fixed_num_events_frames ---")
        return [], [], 0, 0, [], []


    frames, len_arr_x, len_arr_y, center_indices, time_frames_list = [], [], [], [], []
    i, j = 0, 0  # Pointers for positive and negative event arrays

    first_valid_frame_logged = False

    # Loop as long as there are enough events to form a new "packet" for both streams
    while i + window_len_pos <= len(time_data_pos) and j + window_len_neg <= len(time_data_neg):
        current_i, current_j = i, j # Store start indices for this event packet
        
        # Initialize frames for this packet
        current_frame_rgb = np.zeros((img_height, img_width, 3), np.uint8)
        time_frame_raw = np.zeros((img_height, img_width), np.uint8) # For event order, 2D for accumulation

        # Accumulate a fixed number of positive events (window_len_pos)
        num_pos_in_packet = 0
        # The loop condition `idx < current_i + window_len_pos` iterates `window_len_pos` times.
        # `i` is used as the iterator and is advanced.
        end_i_for_packet = current_i + window_len_pos
        while i < end_i_for_packet: # Iterate exactly window_len_pos times
            x = x_data_pos[i]
            y = y_data_pos[i]
            if 0 <= y < img_height and 0 <= x < img_width:
                current_frame_rgb[y][x] = (255, 0, 0)   # Blue for positive
                # Store relative index within this packet for time_frame_raw
                # Value indicates order of arrival within this packet's positive events.
                time_frame_raw[y][x] = max(time_frame_raw[y,x], (i - current_i + 1) % 255) # +1 to avoid 0 for first event
            i += 1
            num_pos_in_packet +=1
        
        # Accumulate a fixed number of negative events (window_len_neg)
        num_neg_in_packet = 0
        end_j_for_packet = current_j + window_len_neg
        while j < end_j_for_packet: # Iterate exactly window_len_neg times
            x = x_data_neg[j]
            y = y_data_neg[j]
            if 0 <= y < img_height and 0 <= x < img_width:
                current_frame_rgb[y][x] = (0, 0, 255)   # Red for negative
                # Store relative index, may overwrite positive event order if at same (x,y) and later
                time_frame_raw[y][x] = max(time_frame_raw[y,x], (j - current_j + 1) % 255) # +1 to avoid 0 for first event
            j += 1
            num_neg_in_packet +=1
        
        # Post-process time_frame_raw (similar to the other function)
        if np.any(time_frame_raw > 0):
            equalized_hist_2d = cv2.equalizeHist(time_frame_raw)
        else:
            equalized_hist_2d = time_frame_raw
        
        active_pixels_mask = equalized_hist_2d > 0
        equalized_hist_2d[active_pixels_mask] = (equalized_hist_2d[active_pixels_mask] / 2).astype(np.uint8) + 50
        final_time_frame = equalized_hist_2d.reshape((img_height, img_width, 1))


        # Calculate extremities and center using events from the current packet
        # `i` and `j` are now the exclusive end indices for the slices processed in this iteration.
        len_x, len_y, overall_x, overall_y = \
            gen_extremities(x_data_pos, y_data_pos, x_data_neg, y_data_neg,
                            current_i, i, current_j, j) 

        # Filter and store frames
        # Condition: at least 50 non-zero pixels in the combined RGB frame (somewhat arbitrary threshold)
        if np.count_nonzero(current_frame_rgb) > 50: # (Original condition, implies current_frame.shape[0/1]>0)
            frames.append(current_frame_rgb)
            len_arr_x.append(len_x)
            len_arr_y.append(len_y)
            center_indices.append((overall_x, overall_y))
            time_frames_list.append(final_time_frame)

            if not first_valid_frame_logged:
                # print(f"DEBUG: First valid frame generated (fixed num events).")
                # print(f"DEBUG:   Positive events in this packet: {num_pos_in_packet} (Target: {window_len_pos}, Index i={i})")
                # print(f"DEBUG:   Negative events in this packet: {num_neg_in_packet} (Target: {window_len_neg}, Index j={j})")
                # print_array_summary("  current_frame_rgb", current_frame_rgb)
                # print_array_summary("  final_time_frame (pseudo-depth)", final_time_frame)
                # print(f"DEBUG:   Calculated extremities: len_x={len_x}, len_y={len_y}, center=({overall_x},{overall_y})")
                # if time_frame_raw.size > 0 : print_array_summary("  Raw time_frame before eq", time_frame_raw)
                first_valid_frame_logged = True
        # else:
            # print(f"DEBUG: Skipped frame (packet) due to insufficient non-zero pixels. Count: {np.count_nonzero(current_frame_rgb)}")

    # --- Loop Summary ---
    # print(f"DEBUG: generate_fixed_num_events_frames main loop finished.")
    # print(f"DEBUG: Total positive events iterated (i): {i} (of {len(time_data_pos)})")
    # print(f"DEBUG: Total negative events iterated (j): {j} (of {len(time_data_neg)})")
    # print(f"DEBUG: Number of valid frames generated: {len(frames)}")

    # Crop the generated frames
    if frames: # Check if any frames were generated to be cropped
        cropped_frames, hx, hy, cropping_positions = \
            generate_cropped_frames(len_arr_x, len_arr_y, frames, center_indices)
        
        crop_width = hx * 2 + 1 if hx > 0 else (1 if len(len_arr_x)>0 and statistics.mean(len_arr_x) == 0 else 0)
        crop_height = hy * 2 + 1 if hy > 0 else (1 if len(len_arr_y)>0 and statistics.mean(len_arr_y) == 0 else 0)
        # print(f"DEBUG: Cropping of frames complete. Target crop_width={crop_width}, crop_height={crop_height} (based on hx={hx}, hy={hy})")
    else:
        # print(f"DEBUG: No frames were generated, so no cropping performed.")
        cropped_frames, cropping_positions = [], []
        hx, hy = 0,0
        crop_width, crop_height = 0,0
    
    # print(f"DEBUG: --- Exiting generate_fixed_num_events_frames ---")
    return frames, cropped_frames, crop_width, crop_height, cropping_positions, time_frames_list