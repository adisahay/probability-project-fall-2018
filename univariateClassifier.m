function accuracy = univariateClassifier(data)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Allow only 1000 x 5 matrices for this case
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[m, n] = size(data);
if m ~= 1000 || n ~= 5
    disp("Expected size of data: 1000 x 5");
    return
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Train the classifier i.e. mean and std/var
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
u = mean(data(1 : 100, :));
s = std(data(1 : 100, :)); % Using std because 'normpdf' requires

p = zeros(1, 5);
pred = zeros(900, 5);

for r = 101 : 1000
    for c = 1 : 5
        for f = 1 : 5
            p(f) = normpdf(data(r, c), u(f), s(f));
        end
        [~, marg] = max(p);
        pred(r - 100, c) = marg;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Find the accuracy. Error rate is 1 - accuracy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
correct = sum(pred == repmat([1 2 3 4 5], 900, 1), 'all');
accuracy = correct / 4500;

end
