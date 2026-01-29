%% POST-PROCESSING ANALYSIS OF OBTAINED DETECTIONS
clear
close all

%% DATA RESHAPING
load dataGT.mat pieces groundTruth

% Circle detections
load resultsVC.mat detections

data = zeros(93, 96);
days = cell(93, 1);
counter = 1;

for i = 1:length(pieces)
    data(i, 1:pieces(i)) = detections(counter:counter + pieces(i) - 1);
    days(i) = {detections(counter:counter + pieces(i) - 1)};
    counter = counter + pieces(i);
end

circle_data = sum(data, 2)';

% Acoustic (KNN)
load resultsKNN_1401.mat detections

data = zeros(93, 96);
days = cell(93, 1);
counter = 1;

for i = 1:length(pieces)
    data(i, 1:pieces(i)) = detections(counter:counter + pieces(i) - 1);
    days(i) = {detections(counter:counter + pieces(i) - 1)};
    counter = counter + pieces(i);
end

% Sum of data per day
acoustic_data_knn = sum(data, 2)';

% Acoustic (SVM)
load resultsSVM_1401.mat detections

data = zeros(93, 96);
days = cell(93, 1);
counter = 1;

for i = 1:length(pieces)
    data(i, 1:pieces(i)) = detections(counter:counter + pieces(i) - 1);
    days(i) = {detections(counter:counter + pieces(i) - 1)};
    counter = counter + pieces(i);
end

% Sum of data per day
acoustic_data_svm = sum(data, 2)';

%% STATISTICAL VALIDATION OF THE DATA
[h1, p1, ci1, stats1] = ttest2(groundTruth, acoustic_data_knn);
c1 = corr2(groundTruth, acoustic_data_knn);

disp('--- KNN vs GroundTruth Analysis ---')
if h1 == 0
    disp('The null hypothesis is not rejected: means are equal')
else
    disp('The null hypothesis is rejected: means are different')
end
disp(['Correlation KNN vs GroundTruth: ', num2str(c1)])
disp(['p-value KNN vs GroundTruth: ', num2str(p1)])

[h2, p2, ci2, stats2] = ttest2(groundTruth, acoustic_data_svm);
c2 = corr2(groundTruth, acoustic_data_svm);

disp('--- SVM vs GroundTruth Analysis ---')
if h2 == 0
    disp('The null hypothesis is not rejected: means are equal')
else
    disp('The null hypothesis is rejected: means are different')
end
disp(['Correlation SVM vs GroundTruth: ', num2str(c2)])
disp(['p-value SVM vs GroundTruth: ', num2str(p2)])

[h3, p3, ci3, stats3] = ttest2(acoustic_data_knn, acoustic_data_svm);
c3 = corr2(acoustic_data_knn, acoustic_data_svm);

disp('--- KNN vs SVM Analysis ---')
if h3 == 0
    disp('The null hypothesis is not rejected: means are equal')
else
    disp('The null hypothesis is rejected: means are different')
end
disp(['Correlation KNN vs SVM: ', num2str(c3)])
disp(['p-value KNN vs SVM: ', num2str(p3)])

%% INITIAL VISUALIZATION
figure
hold on
plot(circle_data, "LineWidth", 1)
plot(acoustic_data_knn, "LineWidth", 1)
plot(acoustic_data_svm, "LineWidth", 1)
plot(groundTruth, "LineWidth", 1)
grid
legend("Circle detection", "KNN", "SVM", "GroundTruth")
xlabel("Days")
ylabel("Detections")
axis([0 93 0 700])
hold off
