function [fitness, best_params] = evaluate_SVMthreshold(threshold, base_route, window, noverlap, nfft, f_min, f_max, func_vals, C_vals, KS_vals)
% Function evaluate_SVMthreshold. Reproduces the feature extraction process
% and returns a fitness value using class balancing and robust validation.
%
% Inputs:
%   threshold: threshold value to be evaluated.
%   base_route: base path where the audio files are located.
%   window: window size for the spectrogram.
%   noverlap: overlap between consecutive windows.
%   nfft: number of FFT points.
%   f_min: minimum frequency to consider.
%   f_max: maximum frequency to consider.
%   func_vals: vector of values for 'KernelFunction' to be tested.
%   C_vals: vector of values for 'BoxConstraint' to be tested.
%   KS_vals: vector of values for 'KernelScale' to be tested.
%
% Additional output:
%   best_params: structure containing the parameters that produced the
%                highest fitness value.

    % Initialize matrices
    X = [];
    y = [];

    % Prepare training files. Audio files from 792 to 796 are used.
    for n = 792:796
        file = sprintf('00000%03d.aif', n);
        route_file = fullfile(base_route, file);
        if ~isfile(route_file)
            continue;
        end

        % Convert to spectrogram and extract energy-based features
        [signal, fs] = audioread(route_file);
        [data, energy_sum, ~, ~, ~] = extract_features( ...
            signal, fs, window, noverlap, nfft, f_min, f_max);

        % Threshold-based labeling
        labels = zeros(size(energy_sum));
        labels(energy_sum > threshold) = 1;
        labels(energy_sum < 0.01 * threshold) = 0;

        % Valid samples: clearly above or below the threshold
        valid_idx = (energy_sum > threshold) | (energy_sum < 0.01 * threshold);

        % Extract features for valid indices
        for i = find(valid_idx)
            X(end+1, :) = [mean(data(:,i)), std(data(:,i)), max(data(:,i))];
            y(end+1) = labels(i);
        end
    end

    % If no valid data are available, return zero fitness
    if isempty(X)
        fitness = 0;
        best_params = [];
        return;
    end

    % Class count
    num1 = sum(y == 1);
    num0 = sum(y == 0);

    % If only one class is present, the fitness is invalid
    if num1 == 0 || num0 == 0
        fitness = 0;
        best_params = [];
        return;
    end

    % Class balancing via random undersampling
    nmin = min(num1, num0);
    idx1 = find(y == 1);
    idx0 = find(y == 0);
    sel1 = randsample(idx1, nmin);
    sel0 = randsample(idx0, nmin);
    sel = [sel1; sel0];

    Xb = X(sel, :);
    yb = double(y(sel(:)));
    yb = yb(:);

    % Safety check for minimum data and class diversity
    if numel(yb) < 10 || numel(unique(yb)) < 2
        fitness = 0;
        best_params = [];
        return;
    end

    % Weights to penalize incorrect detections
    weight_FP = 3;   % False positives
    weight_FN = 1;   % False negatives

    best_fitness = -Inf;
    best_params = struct('Kernel', [], 'C', [], 'KernelScale', []);

    % Grid search over kernel functions and hyperparameters
    for ff = func_vals(:)'
        for C = C_vals
            for ks = KS_vals
                if ff == "rbf"
                    model = fitcsvm(Xb, yb, ...
                        'KernelFunction', ff, ...
                        'BoxConstraint', C, ...
                        'KernelScale', ks, ...
                        'Standardize', true, ...
                        'ClassNames', [0 1]);
                else
                    model = fitcsvm(Xb, yb, ...
                        'KernelFunction', ff, ...
                        'BoxConstraint', C, ...
                        'Standardize', true, ...
                        'ClassNames', [0 1]);
                end

                % Cross-validation and metric computation
                CV = crossval(model, 'KFold', 5);
                ypred = kfoldPredict(CV);
                ytrue = yb(:);

                FP = sum((ytrue == 0) & (ypred == 1));
                FN = sum((ytrue == 1) & (ypred == 0));

                acc = mean(ypred == ytrue);
                cost = FP * weight_FP + FN * weight_FN;
                cost_norm = cost / length(ytrue);

                balance_penalty = 1 - abs(num1 - num0) / (num1 + num0);
                acc_folds = 1 - kfoldLoss(CV, 'mode', 'individual');
                stability_penalty = 1 - std(acc_folds);

                fitness_candidate = ...
                    0.6 * (acc - cost_norm) + ...
                    0.3 * balance_penalty + ...
                    0.1 * stability_penalty;

                if fitness_candidate > best_fitness
                    best_fitness = fitness_candidate;
                    best_params.Kernel = ff;
                    best_params.C = C;
                    best_params.KernelScale = ks;
                end
            end
        end
    end

    % Return the best fitness found
    fitness = best_fitness;
end
