function accuracy = bivariateClassifier(data1, data2)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Allow only 1000 x 5 matrices for this case
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[m1, n1] = size(data1);
if m1 ~= 1000 || n1 ~= 5
    disp("Expected size of data: 1000 x 5");
    return
end
[m2, n2] = size(data2);
if m2 ~= 1000 || n2 ~= 5
    disp("Expected size of data: 1000 x 5");
    return
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Train the classifier i.e. mean and std/var
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
u1 = mean(data1(1 : 100, :));
s1 = std(data1(1 : 100, :)); % Using std because 'normpdf' requires
u2 = mean(data2(1 : 100, :));
s2 = std(data2(1 : 100, :)); % Using std because 'normpdf' requires

p = zeros(1, 5);
pred = zeros(900, 5);

for r = 101 : 1000
    for c = 1 : 5
        for f = 1 : 5
            p(f) = normpdf(data1(r, c), u1(f), s1(f));
            % Assuming independence, we take their products
            p(f) = p(f) * normpdf(data2(r, c), u2(f), s2(f));
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
