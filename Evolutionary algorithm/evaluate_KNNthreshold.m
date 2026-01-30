function [fitness, best_k] = evaluate_KNNthreshold(threshold, base_path, window, noverlap, nfft, f_min, f_max, k_vals)
% Function evaluate_KNNthreshold. Replicates the feature extraction process
% and returns a fitness value with class balancing and safe validation.
%
% Inputs:
%   threshold: threshold value to be evaluated.
%   base_path: base path where the audio files are located.
%   window: window size for the spectrogram.
%   noverlap: overlap between windows.
%   nfft: number of FFT points.
%   f_min: minimum frequency to consider.
%   f_max: maximum frequency to consider.
%   k_vals: vector of values for 'NumNeighbors' to be tested.
%
% Additional outputs:
%   best_k: NumNeighbors value that produced the best fitness.

    % Initialize matrices
    X = [];
    y = [];

    % Prepare files for training (files 792 to 796 are used)
    for n = 792:796
        filename = sprintf('00000%03d.aif', n);
        file_path = fullfile(base_path, filename);
        if ~isfile(file_path)
            continue;
        end

        % Convert to spectrogram and extract features based on energy
        [signal, fs] = audioread(file_path);
        [data, energy_sum, ~, ~, ~] = extract_features( ...
            signal, fs, window, noverlap, nfft, f_min, f_max);

        % Threshold-based labels
        labels = zeros(size(energy_sum));
        labels(energy_sum > threshold) = 1;
        labels(energy_sum < 0.01 * threshold) = 0;
        valid_idx = (labels == 1 | labels == 0);

        % Extract features for valid indices
        for i = find(valid_idx)
            X(end+1, :) = [ ...
                mean(data(:,i)), ...
                std(data(:,i)), ...
                max(data(:,i)) ];
            y(end+1) = labels(i);
        end
    end

    % If no valid data exist, fitness is zero
    if isempty(X)
        fitness = 0;
        best_k = [];
        return;
    end

    % Class counts
    num1 = sum(y == 1);
    num0 = sum(y == 0);

    % If only one class exists, fitness is low (not useful)
    if num1 == 0 || num0 == 0
        fitness = 0;
        best_k = [];
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
    yb = y(sel);

    % Normalize features
    Xb = normalize(Xb);

    % Weights to penalize misdetections
    weight_FP = 3;
    weight_FN = 1;

    % If no valid k values are provided, return zero fitness
    if isempty(k_vals)
        fitness = 0;
        best_k = [];
        return;
    end

    best_fitness = -Inf;
    best_k = [];

    % Grid search over k_vals
    for kk = k_vals(:)'
        model = fitcknn(Xb, yb(:), 'NumNeighbors', kk);
        CV = crossval(model, 'KFold', 5);
        y_pred = kfoldPredict(CV);
        y_true = yb(:);

        FP = sum((y_true == 0) & (y_pred == 1));
        FN = sum((y_true == 1) & (y_pred == 0));

        acc = mean(y_pred == y_true);
        cost = FP * weight_FP + FN * weight_FN;
        cost_norm = cost / length(y_true);

        balance_penalty = 1 - abs(num1 - num0) / (num1 + num0);
        acc_folds = 1 - kfoldLoss(CV, 'mode', 'individual');
        stability_penalty = 1 - std(acc_folds);

        fitness_candidate = ...
            0.6 * (acc - cost_norm) + ...
            0.3 * balance_penalty + ...
            0.1 * stability_penalty;

        % Update best candidate
        if fitness_candidate > best_fitness
            best_fitness = fitness_candidate;
            best_k = kk;
        elseif abs(fitness_candidate - best_fitness) < eps
            % In case of numerical tie, prefer larger k values
            if isempty(best_k) || kk > best_k
                best_fitness = fitness_candidate;
                best_k = kk;
            end
        end
    end

    % Return the best fitness found
    fitness = best_fitness;
end
