close all; clear;

% === General parameters ===
window = 256;
overlap_ratio = (1 - 1/(2^3));
noverlap = floor(overlap_ratio * window);
nfft = window * 8;
f_min = 18;   % Hz
f_max = 24;   % Hz
threshold = 1.787e-16;

% === Paths ===
base_path = '\20110929\';

% === Initialize training variables ===
X_total = [];
Y_total = [];

for n = 792:796
    filename = sprintf('00000%03d.aif', n);
    file_path = fullfile(base_path, filename);

    [signal, fs] = audioread(file_path);

    % Spectrogram
    [~, f, t_spec, s] = spectrogram(signal, window, noverlap, nfft, fs, 'yaxis');
    freq_idx = (f >= f_min) & (f <= f_max);
    data = abs(s(freq_idx, :)).^2;
    data = data - min(data, [], 1);

    % Integrated energy
    energy_sum = sum(data);

    % Labeling
    Y = zeros(1, length(energy_sum));
    Y(energy_sum > threshold) = 1;                 % Whale
    Y(energy_sum < threshold / 10) = 0;            % Background
    valid_idx = (energy_sum > threshold) | ...
                (energy_sum < threshold / 10);     % Only clear data

    % Feature extraction
    X = zeros(length(t_spec), 3);
    for i = 1:length(t_spec)
        X(i,:) = [ ...
            mean(data(:,i)), ...
            std(data(:,i)), ...
            max(data(:,i)) ];
    end

    % Keep only valid labeled data
    X_total = [X_total; X(valid_idx,:)];
    Y_total = [Y_total; Y(valid_idx)'];
end

% === SVM training ===
SVMModel = fitcsvm( ...
    X_total, Y_total, ...
    'KernelFunction', 'rbf', ...
    'Standardize', true, ...
    'ClassNames', [0, 1], ...
    'BoxConstraint', 0.01, ...
    'KernelScale', 0.1);

% === Save model ===
save('SVMModel_1301.mat', 'SVMModel');
disp('SVM classifier trained and saved in SVMModel_1301.mat');
