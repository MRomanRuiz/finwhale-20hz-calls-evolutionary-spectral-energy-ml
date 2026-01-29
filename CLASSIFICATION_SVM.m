close all; clear;

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

% Load trained classifier
load SVMModel_1301 SVMModel

j = 1;
% for n = 797:810
for n = 1:length(audio_ds.Files)
    [signal, fs] = audioread(audio_ds.Files{n});

    % Compute spectrogram
    [~, f, t_spec, s] = spectrogram(signal, window, noverlap, nfft, fs, 'yaxis');
    freq_idx = (f >= f_min) & (f <= f_max);
    data = abs(s(freq_idx, :)).^2;
    data = data - min(data, [], 1);

    % Energy per frame
    energy_sum = sum(data, 1);
    energy_sum = energy_sum(:)'; % row vector

    % Detect frames exceeding threshold
    threshold_idx = energy_sum > threshold;
    threshold_indices = find(threshold_idx);

    if isempty(threshold_indices)
        fprintf('No events detected in file %d\n', n);
        detections(j) = 0;
        j = j + 1;
        continue;
    end

    % Extract features only for those frames
    X = zeros(length(threshold_indices), 3);
    for i = 1:length(threshold_indices)
        ind = threshold_indices(i);
        X(i,:) = [mean(data(:,ind)), std(data(:,ind)), max(data(:,ind))];
    end

    % Classify only those frames
    [pred, ~] = predict(SVMModel, X);

    % Keep only frames classified as whale
    whale_threshold_idx = pred == 1;

    t_whale = t_spec(threshold_indices(pred == 1));

    max_distance = 1;

    if length(t_whale) > 1
        clusters = clusterdata(t_whale', ...
            'linkage', 'single', ...
            'criterion', 'distance', ...
            'cutoff', max_distance);

        cluster_means = accumarray(clusters, t_whale, [], @mean);
        detections(j) = length(cluster_means);
    else
        detections(j) = length(t_whale);
    end

    fprintf('Number of groups found: %d\n', detections(j));

    j = j + 1;
    close all
end
