function [mAP , mAP21] = coco_common_retrieval(I_val_projected , T_val_projected, Z_1_test, Z_2_test, k, ld)
st = tic;

total_columns = size(Z_1_test, 2);
if k == 0
    k = total_columns;
end

% Normalize features
I_val_projected_norm = NormFeat(I_val_projected);
T_val_projected_norm = NormFeat(T_val_projected);

X_1_test = I_val_projected_norm(:,1:ld);
X_2_test = T_val_projected_norm(:,1:ld);

[n_1_t, ~] = size(X_1_test);
[n_2_t, ~] = size(X_2_test);

% Compute similarity matrix in one step
similarity_matrix = exp(-pdist2(X_1_test, X_2_test, 'euclidean').^2);

% Top-k indices for each query in X_1_test
[~, index_matrix] = maxk(similarity_matrix, k, 2);

% Top-k indices for each query in X_2_test (reverse retrieval)
[~, index_matrix2] = maxk(similarity_matrix', k, 2);

% Compute mAP
precision_all = zeros(n_1_t, k);
avg_precision_all = zeros(n_1_t, 1);
for i = 1:n_1_t
    temp = 0;
    count = 0;
    for j = 1:k
        label_similarity = dot(Z_1_test(i,:), Z_2_test(index_matrix(i,j),:)) > 0;
        temp = temp + label_similarity;
        precision_all(i,j) = temp / j;
        if label_similarity
            avg_precision_all(i) = avg_precision_all(i) + precision_all(i,j);
            count = count + 1;
        end
    end
    if count ~= 0
        avg_precision_all(i) = avg_precision_all(i) / count;
    end
end
mAP = mean(avg_precision_all, 'omitnan');

% Compute mAP21
precision_all21 = zeros(n_2_t, k);
avg_precision_all21 = zeros(n_2_t, 1);
for i = 1:n_2_t
    temp = 0;
    count = 0;
    for j = 1:k
        label_similarity = dot(Z_2_test(i,:), Z_1_test(index_matrix2(i,j),:)) > 0;
        temp = temp + label_similarity;
        precision_all21(i,j) = temp / j;
        if label_similarity
            avg_precision_all21(i) = avg_precision_all21(i) + precision_all21(i,j);
            count = count + 1;
        end
    end
    if count ~= 0
        avg_precision_all21(i) = avg_precision_all21(i) / count;
    end
end
mAP21 = mean(avg_precision_all21, 'omitnan');

disp(mAP);
disp(mAP21);
toc(st);
end

