clear; close; clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load the data from the data set file
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dat = load("data.mat");

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Train and test using F1 (Steps 1, 2.1, 2.2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
f1_accuracy = univariateClassifier(dat.F1); % 53% Accuracy
f1_error_rate = 1 - f1_accuracy; % 47% Error Rate

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot F2 vs F1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
hold on;
title("Before normalization of F1");
xlabel("1st Feature (F1)");
ylabel("2nd Feature (F2)");
for i = 1 : 5
    plot(dat.F1(:, i), dat.F2(:, i), 'o');
end
legend('C1', 'C2', 'C3', 'C4', 'C5');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Normalizing F1 rows (Step 3)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dat.Z1 = (dat.F1 - mean(dat.F1, 2)) ./ std(dat.F1, 0, 2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot F2 vs Z1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
hold on;
title("After normalization of F1 to Z1");
xlabel("1st Feature (Z1)");
ylabel("2nd Feature (F2)");
for i = 1 : 5
    plot(dat.Z1(:, i), dat.F2(:, i), 'o');
end
legend('C1', 'C2', 'C3', 'C4', 'C5');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Train and test using Z1 (Step 4, Case 2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
z1_accuracy = univariateClassifier(dat.Z1); % 88.31%% Accuracy
z1_error_rate = 1 - z1_accuracy; % 11.69% Error Rate

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Train and test using F2 (Step 4, Case 3)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
f2_accuracy = univariateClassifier(dat.F2); % 55.09%% Accuracy
f2_error_rate = 1 - f2_accuracy; % 44.91% Error Rate

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Train and test using [Z1, F2](Step 4, Case 4)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
z1f2_accuracy = bivariateClassifier(dat.Z1, dat.F2); % 97.98% Accuracy
z1f2_error_rate = 1 - z1f2_accuracy; % 2.02% Error Rate
