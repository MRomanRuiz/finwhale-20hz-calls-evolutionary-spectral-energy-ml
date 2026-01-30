function knnModel = trainKNN( ...
    basePath, fileRange, windowSize, overlapRatio, ...
    nfft, fMin, fMax, threshold, k)
% Function trainKNN
% Trains a k-NN classifier for acoustic detection.
%
% Inputs:
%   basePath      : path where the audio files are located
%   fileRange     : range of file indices to process [start end]
%   windowSize    : window size for the spectrogram
%   overlapRatio  : overlap ratio between consecutive windows
%   nfft          : number of FFT points
%   fMin          : minimum frequency to consider
%   fMax          : maximum frequency to consider
%   threshold     : energy threshold for detection
%   k             : number of neighbors for the k-NN classifier
%
% Output:
%   knnModel      : trained k-NN model

    % Initial message
    fprintf('\nTraining KNN model with threshold = %.3e\n', threshold);

    % Derived parameters
    noverlap = floor(overlapRatio * windowSize);

    % Initialize feature and label matrices
    features = [];
    labels = [];

    % Prepare files for training
    for fileIdx = fileRange(1):fileRange(2)
        fileName = sprintf('00000%03d.aif', fileIdx);
        filePath = fullfile(basePath, fileName);

        if ~isfile(filePath)
            fprintf('File not found: %s\n', fileName);
            continue;
        end

        % Read audio and extract energy-based features
        [audioSignal, fs] = audioread(filePath);
        [specData, energySum] = extract_features( ...
            audioSignal, fs, windowSize, noverlap, ...
            nfft, fMin, fMax);

        % Threshold-based labeling
        sampleLabels = zeros(size(energySum));
        sampleLabels(energySum > threshold) = 1;          % whale
        sampleLabels(energySum < 0.01 * threshold) = 0;   % background

        validIdx = (sampleLabels == 1 | sampleLabels == 0);

        % Extract features for valid samples
        for i = find(validIdx)
            features(end+1, :) = [ ...
                mean(specData(:,i)), ...
                std(specData(:,i)), ...
                max(specData(:,i)) ];
            labels(end+1) = sampleLabels(i);
        end
    end

    % Check for valid data
    if isempty(features)
        error('No valid data available to train the model');
    end

    % Class count
    numWhale = sum(labels == 1);
    numBackground = sum(labels == 0);

    if numWhale == 0 || numBackground == 0
        error('Samples from both classes are required to train the model');
    end

    % Class balancing via random undersampling
    minSamples = min(numWhale, numBackground);
    whaleIdx = find(labels == 1);
    backgroundIdx = find(labels == 0);

    selectedWhale = randsample(whaleIdx, minSamples);
    selectedBackground = randsample(backgroundIdx, minSamples);
    selectedIdx = [selectedWhale; selectedBackground];

    balancedFeatures = features(selectedIdx, :);
    balancedLabels = labels(selectedIdx);
    balancedLabels = balancedLabels(:);

    % Normalize features
    balancedFeatures = normalize(balancedFeatures);

    % Train k-NN model
    knnModel = fitcknn(balancedFeatures, balancedLabels, ...
                       'NumNeighbors', k);

end
