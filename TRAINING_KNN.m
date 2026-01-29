close all; clear;

% Parameters
window = 256;
overlap_ratio = (1 - 1/(2^3));
noverlap = floor(overlap_ratio * window);
nfft = window * 8;
f_min = 18;
f_max = 24;
threshold = 8.403e-16;

% Number of neighbors
k = 72;

% Base path
base_path = '\20110929\';

% Initialize matrices
X = [];
y = [];

% Files used for training
for n = 792:796
    filename = sprintf('00000%03d.aif', n);
    file_path = fullfile(base_path, filename);

    if ~isfile(file_path)
        fprintf('Not found: %s\n', filename);
        continue;
    end

    [signal, fs] = audioread(file_path);
    [~, f, t_spec, s] = spectrogram(signal, window, noverlap, nfft, fs, 'yaxis');

    freq_idx = (f >= f_min) & (f <= f_max);
    data = abs(s(freq_idx, :)).^2;
    data = data - min(data, [], 1);

    energy_sum = sum(data, 1);
    energy_sum = energy_sum(:)';

    % Threshold-based labels
    labels = zeros(size(energy_sum));
    labels(energy_sum > threshold) = 1;            % whale
    labels(energy_sum < 0.01 * threshold) = 0;     % background

    valid_idx = (labels == 1 | labels == 0);

    for i = find(valid_idx)
        X(end+1, :) = [ ...
            mean(data(:,i)), ...
            std(data(:,i)), ...
            max(data(:,i)) ];
        y(end+1) = labels(i);
    end
end

% Train k-NN classifier
knn_model = fitcknn(X, y', 'NumNeighbors', k);

% Save model
save('KNNModel_0801.mat', 'knn_model');
disp('KNN model saved to KNNModel_0801.mat');
