close all; clear;

tic

% Parameters
window = 256;
overlap_ratio = (1 - 1/(2^3));
noverlap = floor(overlap_ratio * window);
nfft = window * 8;
f_min = 18;
f_max = 24;
threshold = 1e-15;

% Base path
base_path = 'D:\EAR COL Sept2011\EAR6_COL_Sept2011 AIFF\EAR_COL_Sept2011 250Hz AIFF\';

audio_ds = audioDatastore(base_path);

% Load trained k-NN model
load KNNModel_0801.mat knn_model

% Classify new files
j = 1;

for n = 1:length(audio_ds.Files)
    [signal, fs] = audioread(audio_ds.Files{n});
    [~, f, t_spec, s] = spectrogram(signal, window, noverlap, nfft, fs, 'yaxis');

    freq_idx = (f >= f_min) & (f <= f_max);
    data = abs(s(freq_idx, :)).^2;
    data = data - min(data, [], 1);

    energy_sum = sum(data, 1);
    energy_sum = energy_sum(:)';

    % Threshold-based detection
    threshold_idx = energy_sum > threshold;
    threshold_indices = find(threshold_idx);

    if isempty(threshold_indices)
        fprintf('No events detected in file %d\n', n);
        detections(j) = 0;
        j = j + 1;
        continue;
    end

    % Extract features only for detected events
    X_test = zeros(length(threshold_indices), 3);
    for i = 1:length(threshold_indices)
        ind = threshold_indices(i);
        X_test(i,:) = [mean(data(:,ind)), std(data(:,ind)), max(data(:,ind))];
    end

    % k-NN classification
    predictions = predict(knn_model, X_test);

    whale_times = t_spec(threshold_indices(predictions == 1));

    max_distance = 1;

    if length(whale_times) > 1
        clusters = clusterdata(whale_times', ...
            'linkage', 'single', ...
            'criterion', 'distance', ...
            'cutoff', max_distance);

        cluster_means = accumarray(clusters, whale_times, [], @mean);
        detections(j) = length(cluster_means);
    else
        detections(j) = length(whale_times);
    end

    fprintf('Number of groups found: %d\n', detections(j));

    j = j + 1;
    close all
end

toc
