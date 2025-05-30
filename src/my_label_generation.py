import cv2
import numpy as np

from src.my_visualize import *
from src.my_util import *
from src.my_event_frame_generation import *
from mask_matching import get_mask_for_index_and_label


def create_gray_summed_mat(cropped_frames, len_y, len_x):
    gray_summed_mat = cv2.cvtColor(np.zeros((len_y, len_x, 3), np.uint8), cv2.COLOR_BGR2GRAY)

    for cropped in cropped_frames:
        # print(cropped.shape)
        if cropped.shape[0] > 0 and cropped.shape[1] > 0:
            gray_scaled = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            x_arr, y_arr = np.where(gray_scaled > 0)
            for ind, x in enumerate(x_arr):
                y = y_arr[ind]
                gray_summed_mat[x][y] += 1

    return gray_summed_mat


def color_yellow(image):
    x, y = np.where(image > 0)
    colorized_mask = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    colorized_mask[x, y] = (0, 255, 255, 255)  # YELLOW
    return colorized_mask


def generate_average_mask(stretched_mat):
    resized = resize_image(stretched_mat, 2000)
    kernel_size = 11
    closing_gray_summed_mat = cv2.morphologyEx(resized, cv2.MORPH_CLOSE,
                                               make_kernel(kernel_size)[0], iterations=3, borderType=cv2.BORDER_CONSTANT)
    (thresh, th_gray_summed_mat) = cv2.threshold(closing_gray_summed_mat, 100, 255, cv2.THRESH_BINARY)  # 36
    revert_resize = resize_image(th_gray_summed_mat, 5)
    colorized_mask = color_yellow(revert_resize)

    # show_partial_matrices(closing_gray_summed_mat, stretched_mat, th_gray_summed_mat, revert_resize, colorized_mask)

    return colorized_mask


dx = [-1, 0, 1, 1, 1, 0, -1, -1,
          -2, -1, 0, 1, 2, 2, 2, 2, 2, 1, 0, -1, -2, -2, -2, -2,
          -3, -2, -1, 0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -3, -3, -3]
dy = [-1, -1, -1, 0, 1, 1, 1, 0,
      -2, -2, -2, -2, -2, -1, 0, 1, 2, 2, 2, 2, 2, 1, 0, -1,
      -3, -3, -3, -3, -3, -3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 2, 1, 0, -1, -3]


def mask_placement_correction(y0, y1, x0, x1, mask, frame, bad_frame):
    global mask_scores
    initial_max_score = calculate_mask_placement_score(frame, mask, x0, x1, y0, y1)
    max_score = initial_max_score

    y0_mx, y1_mx, x0_mx, x1_mx = y0, y1, x0, x1

    for d in range(len(dx)):
        x0_new, y0_new = x0 + dx[d], y0 + dy[d]
        x1_new, y1_new = x1 + dx[d], y1 + dy[d]
        if x0_new < 0 or y0_new < 0 or x1_new >= frame.shape[1] or y1_new >= frame.shape[0]:
            continue
        if mask_scores[x0_new][y0_new] > 0:
            continue
        position_bw_mask_new = np.zeros(frame.shape, np.uint8)
        position_bw_mask_new[y0_new:y1_new, x0_new:x1_new] = mask[0:y1_new-y0_new, 0:x1_new-x0_new]
        combined_new = cv2.bitwise_and(frame, position_bw_mask_new)
        #bad_combined_new = cv2.bitwise_and(bad_frame, position_bw_mask_new)
        current_score = np.count_nonzero(combined_new) #- np.count_nonzero(bad_combined_new)
        mask_scores[x0_new][y0_new] = current_score
        if current_score > max_score:
            max_score = current_score
            y0_mx, y1_mx, x0_mx, x1_mx = y0_new, y1_new, x0_new, x1_new
            # print("new max score:", current_score)
            # print(y0_new, y1_new, x0_new, x1_new)
            # cv2.imshow('combined_new', combined_new)
            # cv2.waitKey(200)

    if max_score > 0:
        score_difference = (max_score-initial_max_score)/max_score
        # print('scores:', max_score, initial_max_score, score_difference)
        if score_difference > 0.02:
            y0_mx, y1_mx, x0_mx, x1_mx, max_score = \
                mask_placement_correction(y0_mx, y1_mx, x0_mx, x1_mx, mask, frame, bad_frame)

    return y0_mx, y1_mx, x0_mx, x1_mx, max_score


def calculate_mask_placement_score(frame, mask, x0, x1, y0, y1):
    position_bw_mask = np.zeros(frame.shape, np.uint8)
    position_bw_mask[y0:y1, x0:x1] = mask[0:y1-y0, 0:x1-x0]
    combined = cv2.bitwise_and(frame, position_bw_mask)
    return np.count_nonzero(combined)


def place_mask(mask, frame, y0, y1, x0, x1):
    y0_mx, y1_mx, x0_mx, x1_mx = y0, y1, x0, x1
    red_score = calculate_mask_placement_score(frame[:, :, 2], mask, x0, x1, y0, y1)  # Red
    blue_score = calculate_mask_placement_score(frame[:, :, 0], mask, x0, x1, y0, y1)  # Blue
    mx_score = red_score + int(blue_score / 10)

    hh, ww = mask.shape

    for y0 in range(hh):
        for x0 in range(ww):
            y1, x1 = y0+hh, x0+ww
            if x1 >= frame.shape[1] or y1 >= frame.shape[0]:
                continue
            red_current_score = calculate_mask_placement_score(frame[:, :, 2], mask, x0, x1, y0, y1)  # Red
            blue_current_score = calculate_mask_placement_score(frame[:, :, 0], mask, x0, x1, y0, y1)  # Blue
            current_score = red_current_score + int(blue_current_score / 10)
            mask_scores[x0][y0] = current_score
            if current_score > mx_score:
                mx_score = current_score
                y0_mx, y1_mx, x0_mx, x1_mx = y0, y1, x0, x1

    return y0_mx, y1_mx, x0_mx, x1_mx, mx_score


def generate_masks(dataset_entry, index, last_saved_index, mask_indices_per_label, mnist_dataset, **kwargs):
    global mask_scores
    my_events, target = dataset_entry
    print("Target:", target)

    window_len = kwargs.get('window_len', 20)

    positive_event_array = generate_event_arrays(my_events, 1)
    negative_event_array = generate_event_arrays(my_events, 0)

    denoise_transform = tonic.transforms.Denoise(filter_time=5000)
    events_denoised = denoise_transform(my_events)
    # events_denoised = my_events

    positive_event_array_denoised = generate_event_arrays(events_denoised, 1)
    negative_event_array_denoised = generate_event_arrays(events_denoised, 0)


    # TODO Turn this into generation of fixed window length - DONE

    # frames, cropped_frames, len_x, len_y, cropping_positions, time_frames \
        # = generate_fixed_num_events_frames(positive_event_array, negative_event_array)

    frames, frames_denoised, cropped_frames, len_x, len_y, cropping_positions, time_frames \
        = generate_event_frames_with_fixed_time_window(positive_event_array_denoised, negative_event_array_denoised,
                                                       positive_event_array, negative_event_array, window_len)

    # Get mask based on index and target

    # TODO: replace this with correct mask - DONEE
    # gray_summed_mat = create_gray_summed_mat(cropped_frames, len_y, len_x)
    # stretched_mat = contrast_stretch(gray_summed_mat, len(cropped_frames))
    # colorized_mask = generate_average_mask(stretched_mat)
    correct_mask = get_mask_for_index_and_label(index - last_saved_index, target,
                                                mnist_dataset, mask_indices_per_label)
    colorized_mask = color_yellow(correct_mask) # TODO Fix this
    # cv2.imshow('correct_mask', correct_mask)
    # cv2.imshow('colorized_mask', colorized_mask)
    # cv2.waitKey(500)
    # print('Colorized', colorized_mask.shape)

    label_masked_frames = []
    colorized_masks = []
    for ind, frame in enumerate(frames):
        frame_denoised = frames_denoised[ind]
        # print(np.count_nonzero(frame))
        (y0, y1, x0, x1) = cropping_positions[ind]
        result = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        positioned_colorized_mask = np.zeros(result.shape, np.uint8)
        new_colorized_mask = np.zeros(result.shape, np.uint8)
        phh, pww, pcc = positioned_colorized_mask.shape
        hh, ww, cc = colorized_mask.shape
        # print(y1, phh, x1, pww)
        hh -= max(y1 - phh, 0)
        ww -= max(x1 - pww, 0)
        hh = min(hh, phh - y0)
        ww = min(ww, pww - x0)
        # print(hh, ww, y1-y0, x1-x0)
        # print(x0, x0+ww)

        mask_scores = np.zeros((phh, pww, 1), np.uint8)
        # cv2.imshow('frame_denoised', frame_denoised)
        # cv2.imshow('red', frame_denoised[:, :, 2])
        # cv2.imshow('blue', frame_denoised[:, :, 0])

        y0, y1, x0, x1, mx_score = place_mask(correct_mask, frame_denoised,
                                              y0, (y0+hh), x0, (x0+ww))

        y0, y1, x0, x1, mx_score = \
            mask_placement_correction(y0, (y0+hh), x0, (x0+ww), correct_mask,
                                      frame_denoised[:, :, 2], frame_denoised[:, :, 0])

        new_colorized_mask[y0:y1, x0:x1] = colorized_mask[0:hh, 0:ww]
        new_result = cv2.addWeighted(result, 1, new_colorized_mask, 0.5, 0)

        equalized_hist = np.array(cv2.equalizeHist(mask_scores))
        x_pixels, y_pixels = np.where(equalized_hist > 0)
        equalized_hist[x_pixels, y_pixels] = equalized_hist[x_pixels, y_pixels]

        # mask_scores = cv2.equalizeHist(mask_scores)
        # cv2.imshow('mask_scores', equalized_hist)
        # cv2.imshow('new_result', new_result)
        # cv2.waitKey(0)

        positioned_colorized_mask[y0:y1, x0:x1] = colorized_mask[0:hh, 0:ww]

        result = cv2.addWeighted(result, 1, positioned_colorized_mask, 0.5, 0)

        colorized_masks.append(positioned_colorized_mask)
        label_masked_frames.append(result)

        # cv2.imshow('neg_frame', frame[:, :, 2])
        # cv2.imshow('positioned_colorized_mask', positioned_colorized_mask)

        # print(frame[:, :, 2].shape)
        # print(position_bw_mask.shape)
        # cv2.imshow('masked', result)
        # cv2.imshow('new_result', new_result)
        # cv2.waitKey(500)

    # show_events(colorized_masks, 'colorized_masks/frame_' + str(target) + '_')
    return frames, colorized_masks, target, time_frames


