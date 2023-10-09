
import numpy as np
import pandas as pd
import numpy.random as npr
import matplotlib.pyplot as plt
import scipy.io

import tensorflow as tf 
from keras.layers import Dense, LSTM, BatchNormalization, Bidirectional
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.regularizers import l2
from tensorflow.keras import regularizers


class BehaviorAnalysis:
    def __init__(self):
        pass

    def structured_array_to_dict(self, arr):
        if not arr.dtype.names:
            return arr.tolist()
        return {name: self.structured_array_to_dict(arr[name]) for name in arr.dtype.names}

    def convert_dataCell_to_dict(self, mat_Cell_data):
        dataCell_list = [self.structured_array_to_dict(mat_Cell_data['dataCell'][0, i][0, 0]) for i in range(mat_Cell_data['dataCell'].shape[1])]
        return dataCell_list
    
    def convert_mov_data_to_dict(self, mat_data, dataCell_list):
        # Extract y-position data
        y_position = mat_data['data'][2]

        # Identify transitions from ~300 to 0 as trial end points
        trial_end_indices = np.where((y_position[:-1] > 250) & (y_position[1:] < 50))[0]

        # Verify the number of trials
        number_of_identified_trials = len(trial_end_indices)
        number_of_trials_in_dataCell = len(dataCell_list)

        number_of_identified_trials, number_of_trials_in_dataCell

        # Segment the data based on trial end indices to define data_segments
        data_segments = []
        start_idx = 0
        for end_idx in trial_end_indices:
            segment = mat_data['data'][:, start_idx:end_idx+1]
            data_segments.append(segment)
            start_idx = end_idx + 1

        # Find the length of the longest trial for padding
        max_trial_length = max([segment.shape[1] for segment in data_segments])

        # Pad each trial segment with NaN values to match the length of the longest trial
        padded_data_segments = []
        for segment in data_segments:
            padding_length = max_trial_length - segment.shape[1]
            padded_segment = np.pad(segment, ((0, 0), (0, padding_length)), constant_values=np.nan)
            padded_data_segments.append(padded_segment)

        # Convert the list of padded segments into a 3D numpy array
        padded_data_matrix_3D = np.stack(padded_data_segments, axis=0)

        keys = ["time", "x-position", "y-position", "view angle", "x velocity", "y velocity", "current world", "reward", "iti"]

        # Create the dictionary with specified keys and 2D matrices for each feature
        mov_data_list = {key: padded_data_matrix_3D[:, idx, :] for idx, key in enumerate(keys)}

        return mov_data_list

    def get_perc_correct_all_conditions(self, dataCell_list):
        correct_values = [int(entry['result']['correct'][0][0][0]) for entry in dataCell_list]
        condition_values = [int(entry['maze']['condition'][0][0][0]) for entry in dataCell_list]
        total_counts = {}
        correct_counts = {}
        for condition, correct in zip(condition_values, correct_values):
            if condition not in total_counts:
                total_counts[condition] = 0
                correct_counts[condition] = 0
            total_counts[condition] += 1
            correct_counts[condition] += correct
        percentage_correct = {condition: (correct_counts[condition] / total_counts[condition]) * 100 for condition in total_counts}
        sorted_conditions = sorted(percentage_correct.keys())
        sorted_percentages = [percentage_correct[condition] for condition in sorted_conditions]
        return correct_values, condition_values, total_counts, correct_counts, percentage_correct, sorted_conditions, sorted_percentages

    def running_mean(self, data, window_size):
        cumsum = np.cumsum(data)
        cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
        return cumsum[window_size - 1:] / window_size

    def determine_color(self, condition):
        if condition in [1, 4, 5, 8]:
            return "pink"
        elif condition in [2, 3, 6, 7]:
            return "orange"
        else:
            return "gray"

    def plot_perc_correct_all_conditions(self, sorted_conditions, sorted_percentages):
        fig, ax = plt.subplots(figsize=(5, 3))
        colors = ['red' if condition <= 4 else 'blue' for condition in sorted_conditions]
        bars = ax.bar(sorted_conditions, sorted_percentages, color=colors, alpha=0.7)
        ax.set_xlabel('Condition')
        ax.set_ylabel('% Correct')
        ax.set_title('Percentage of Correct Choices for Each Condition')
        ax.set_ylim(0, 100)
        ax.set_xticks(sorted_conditions)
        ax.legend(handles=[bars[0], bars[4]], labels=['V-relevant', 'A-relevant'])
        plt.tight_layout()
        plt.show()

    def plot_performance(self, condition_values, correct_values):
        trial_colors = [self.determine_color(condition) for condition in condition_values]
        shift_points = []
        for i in range(1, len(condition_values)):
            if (condition_values[i-1] <= 4 and condition_values[i] >= 5) or (condition_values[i-1] >= 5 and condition_values[i] <= 4):
                shift_points.append(i)
        window_size = 20
        smoothed_values = self.running_mean(correct_values, window_size)
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 5))
        start_idx = window_size - 1
        for i in range(start_idx, len(correct_values)):
            color = "red" if condition_values[i] <= 4 else "blue"
            ax1.plot([i-1, i], [smoothed_values[i-start_idx-1], smoothed_values[i-start_idx]], color=color, label='_nolegend_')
        legend_labels = ["V-rel", "A-rel"]
        legend_colors = ["red", "blue"]
        ax1.axhline(y=0.7, color='black', linestyle='--', linewidth=1)
        markers = [plt.Line2D([0,0], [0,0], color=color, marker='o', linestyle='') for color in legend_colors]
        ax1.legend(markers, legend_labels, numpoints=1)
        ax1.set_title("Running Mean of Correct Choices")
        for shift_point in shift_points:
            ax1.axvline(x=shift_point, color='black', linestyle='--', linewidth=1)
        ax1.set_xlim(0,400)
        ax1.set_ylim(0,1)
        jitter_amount = 0.05
        jittered_correct_values = [value + np.random.uniform(-jitter_amount, jitter_amount) for value in correct_values]
        scatter = ax2.scatter(range(len(jittered_correct_values)), jittered_correct_values, c=trial_colors, alpha=0.7, label='_nolegend_')
        for shift_point in shift_points:
            ax2.axvline(x=shift_point, color='black', linestyle='--', linewidth=1)
        legend_labels = ["Congruent", "Incongruent"]
        legend_colors = ["pink", "orange"]
        markers = [plt.Line2D([0,0], [0,0], color=color, marker='o', linestyle='') for color in legend_colors]
        ax2.legend(markers, legend_labels, numpoints=1)
        ax2.set_title("Scatter Plot of Correct vs. Incorrect Trials")
        ax2.set_xlabel("Trial")
        ax2.set_ylabel("Correct (1) / Incorrect (0)")
        ax2.set_xlim(0,400)
        ax2.set_yticks([0,1])
        ax2.set_yticklabels(['0','1'])
        plt.tight_layout()
        plt.show()

class LSTM_dynamic_choice:
    def __init__(self):
        pass

    def prepare_data_for_LSTM(self, mov_data_list, correct_labels, desired_length):
        # Prepare the input data
        X = np.stack([mov_data_list['x-position'], mov_data_list['y-position'], mov_data_list['x velocity'], 
                    mov_data_list['y velocity'], mov_data_list['view angle']], axis=2)

        # Use the 'Correct' column as the target (reported choice)
        y = np.array(correct_labels)

        X_cleaned = [trial[~np.isnan(trial).any(axis=1)] for trial in X]

        # Apply linear subsampling to each trial
        X_subsampled = [LSTM_dynamic_choice.linear_subsample(trial, desired_length) for trial in X_cleaned]

        # Convert list of arrays into a 3D numpy array
        X_subsampled_array = np.stack(X_subsampled, axis=1)

        # Correctly replicate the y labels for each of the 200 subsampled time points
        y_repeated = np.tile(y.reshape(1, -1), (desired_length, 1))

        return X_subsampled_array, y_repeated
    
    def linear_subsample(trial_data, desired_length):
        
        trial_length = trial_data.shape[0]
        
        if trial_length <= desired_length:
            return trial_data
        else:
            # Select 200 evenly spaced indices from the trial data
            indices = np.linspace(0, trial_length-1, desired_length).astype(int)
            return trial_data[indices]

    def lstm_model(self, input_shape, regularization=0.01):
        model = Sequential()
        model.add(Bidirectional(LSTM(10, input_shape=input_shape, return_sequences=True)))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=.01), metrics=['accuracy'])
        return model
    

