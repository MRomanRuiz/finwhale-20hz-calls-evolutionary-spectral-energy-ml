function [data, energy_sum, f, t_spec, freq_idx] = extract_features(signal, fs, window, noverlap, nfft, f_min, f_max)
% Function extract_features. Extracts features from an audio signal using
% the spectrogram.
%
% Inputs:
%   signal: audio signal to be processed
%   fs: sampling frequency
%   window: window size for the spectrogram
%   noverlap: overlap between windows
%   nfft: number of FFT points
%   f_min: minimum frequency to consider
%   f_max: maximum frequency to consider
%
% Outputs:
%   data: matrix containing the filtered and processed spectrogram data
%   energy_sum: vector containing the energy sum per column
%   f: frequency vector
%   t_spec: time vector
%   freq_idx: logical mask indicating which frequencies fall within
%             the range [f_min, f_max]

    % Compute spectrogram
    [~, f, t_spec, s] = spectrogram(signal, window, noverlap, nfft, fs, 'yaxis');

    % Filter by frequency range
    freq_idx = (f >= f_min) & (f <= f_max);

    % Process data
    data = abs(s(freq_idx, :)).^2;
    data = data - min(data, [], 1);  % subtract column-wise minimum

    % Compute energy sum
    energy_sum = sum(data, 1);
    energy_sum = energy_sum(:)';  % ensure row vector
end
