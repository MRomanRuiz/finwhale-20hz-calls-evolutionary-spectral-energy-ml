%% THRESHOLD TESTING
% Code to test the threshold computed using the evolutionary approach,
% comparing energy-based detection and k-NN classification.
close all; clear; clc;

%% File loading and parameters
% Load test file
audio_dir = 'EAR_COL_Sept2011 250Hz AIFF\';
ADS = audioDatastore(audio_dir);
file_idx = 796;
[audio_signal, fs] = audioread(ADS.Files{file_idx});

% Spectrogram parameters
window_size = 256;
overlap_ratio = (1 - 1/(2^3));
noverlap = floor(overlap_ratio * window_size);
nfft = window_size * 8;
f_min = 18;
f_max = 24;

% Previously computed optimal threshold and classifier info
threshold = 1.794e-16;
base_path = '20110929\';
training_files = [792, 796];

classifier_type = 'trainKNN';
num_neighbors = 72;
kernel_function = 'rbf';
box_constraint = 0.01;
kernel_scale = 0.1;

%% Energy-based processing
% Feature extraction
[spec_data, energy_sum, freq, time_spec, freq_idx] = ...
    extract_features(audio_signal, fs, window_size, ...
                     noverlap, nfft, f_min, f_max);

% Threshold-based detection
threshold_mask = energy_sum > threshold;
threshold_indices = find(threshold_mask);

if isempty(threshold_indices)
    fprintf('No events detected\n');
    detection_result = 0;
end

% Feature extraction for detected events only
X_test = zeros(length(threshold_indices), 3);
for i = 1:length(threshold_indices)
    idx = threshold_indices(i);
    X_test(i,:) = [mean(spec_data(:,idx)), ...
                   std(spec_data(:,idx)), ...
                   max(spec_data(:,idx))];
end

%% Training and classification
switch classifier_type
    case "trainKNN"
        knn_model = trainKNN(base_path, training_files, ...
            window_size, overlap_ratio, nfft, ...
            f_min, f_max, threshold, num_neighbors);
        predictions = predict(knn_model, X_test);

    case "trainSVM"
        svm_model = trainSVM(base_path, training_files, ...
            window_size, overlap_ratio, nfft, ...
            f_min, f_max, threshold, ...
            kernel_function, box_constraint, kernel_scale);
        predictions = predict(svm_model, X_test);
end

%% Visualization
figure('Name', "Acoustic Analysis");

% Spectrogram
subplot(3,1,1);
imagesc(time_spec, freq(freq_idx), spec_data);
axis xy; colormap(jet);
title('Spectrogram');
xlabel('Time (s)');
ylabel('Frequency (Hz)');

% Threshold-based detection
subplot(3,1,2);
plot(time_spec, energy_sum); hold on;
plot(time_spec(threshold_indices), ...
     energy_sum(threshold_indices), 'ro', ...
     'MarkerFaceColor', 'r');
title('Energy threshold detection');
ylabel('Energy');
axis tight;

% Classification results (whale vs background)
subplot(3,1,3);
plot(time_spec, zeros(size(time_spec)), 'k.'); hold on;
whale_times = time_spec(threshold_indices(predictions == 1));
plot(whale_times, 1.1 * ones(size(whale_times)), ...
     'go', 'MarkerFaceColor', 'g');
ylim([-0.5 1.5]);
axis tight;
title('Events classified as whale');
xlabel('Time (s)');
ylabel('Label');

drawnow;

%% Temporal clustering of detections
max_distance = 1;

if length(whale_times) > 1
    cluster_ids = clusterdata(whale_times', ...
        'linkage', 'single', ...
        'criterion', 'distance', ...
        'cutoff', max_distance);
    detection_result = accumarray(cluster_ids, ...
        whale_times, [], @mean);
end

%% Final results
fprintf('Number of clusters found: %d\n', length(detection_result));

% Indices classified as whale
whale_indices = threshold_indices(predictions == 1);

% Mean spectral energy of whale detections
mean_spectral_energy = mean(energy_sum(whale_indices));
fprintf('Mean spectral energy (label 1): %.3e\n', mean_spectral_energy);
