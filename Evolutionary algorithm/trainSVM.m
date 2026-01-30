function svmModel = trainSVM( ...
    basePath, fileRange, windowSize, overlapRatio, ...
    nfft, fMin, fMax, threshold, kernelFunc, C, kernelScale)
% Function trainSVM
% Trains an SVM classifier for acoustic detection.
%
% Inputs:
%   basePath      : path where the audio files are located
%   fileRange     : range of file indices to process [start end]
%   windowSize    : window size for the spectrogram
%   overlapRatio  : overlap between consecutive windows
%   nfft          : number of FFT points
%   fMin          : minimum frequency to consider
%   fMax          : maximum frequency to consider
%   threshold     : energy threshold for detection
%   kernelFunc    : kernel function to use
%   C             : BoxConstraint for SVM
%   kernelScale   : KernelScale for SVM (only if kernelFunc is 'rbf')
%
% Output:
%   svmModel      : trained SVM model

    % Initial message
    fprintf('\nTraining SVM model with threshold = %.3e\n', threshold);

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

        validIdx = (energySum > threshold) | ...
                   (energySum < 0.01 * threshold);

        % Extract features for valid samples
        for i = find(validIdx)
            features(end+1, :) = [ ...
                mean(specData(:,i)), ...
                std(specData(:,i)), ...
                max(specData(:,i)) ];
            labels(end+1,1) = sampleLabels(i);
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
    balancedLabels = double(labels(selectedIdx));
    balancedLabels = balancedLabels(:);

    % Train SVM model
    if kernelFunc == "rbf"
        svmModel = fitcsvm( ...
            balancedFeatures, balancedLabels, ...
            'KernelFunction', kernelFunc, ...
            'BoxConstraint', C, ...
            'KernelScale', kernelScale, ...
            'Standardize', true, ...
            'ClassNames', [0 1]);
    else
        svmModel = fitcsvm( ...
            balancedFeatures, balancedLabels, ...
            'KernelFunction', kernelFunc, ...
            'BoxConstraint', C, ...
            'Standardize', true, ...
            'ClassNames', [0 1]);
    end

end
