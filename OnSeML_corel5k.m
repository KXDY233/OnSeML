clear all; close all; clc;
addpath(genpath('./Tools'));
addpath('./Predict');
 % addpath(genpath('./SuiteSparse'));
 
profile on

%% Data processing 
para.filename = 'Corel5k'; % bibtex delicious
[dataX, dataY] = load_data(para.filename);
saveX = dataX;
saveY = dataY;

dataY = full(dataY);
% dataY = dataY' ./ sum(dataY');
% dataY = dataY';
dataY(dataY == 0) = -1;

f_d = size(dataX,1); % fea_dimension
l_d = size(dataY,1); % label_dimension
num_data = size(dataX,2);

para.num_batch = 20; 
para.plabelled = 0.2;
para.low_dim = 10;
para.lambda = 0.001;

para.tk = 5; % number of predicted labels

options = [];
options.WeightMode = 'HeatKernel'; % Cosine HeatKernel
options.bSelfConnected = 1;
options.k = 0;
 
% the first batch is full-labelled; while the others are randomly labelled
 
% indices is the batch number index
indices = zeros(num_data,1);
size_batch  = round(num_data / para.num_batch);
for i=1:para.num_batch
    start_p = 1 + (i-1) * size_batch;
    end_p = i * size_batch;
    if end_p < num_data
        indices(start_p:end_p) = i;
    else
        indices(start_p:end) = i;
    end
end 
 
% random initialization
 
size_ini = size(indices(indices == 1), 1);  
num_labelled_b = round(num_data * para.plabelled) - size_ini;
num_unlabelled = num_data - round(num_data * para.plabelled);
 
unlabel_index = size_ini + randperm(num_data - size_ini, num_unlabelled);
dataY(:,unlabel_index) = -1;
 
% label_index place a flag of labelled/unlabelled
label_index = ones(num_data, 1);
label_index(unlabel_index) = 0;
 
% generate an orthogonalized encoding matrix P_0
% get the related decoding matrix Q_0 (with H_0 and K_00)
Pcell = cell(num_data,1);
Hcell = cell(num_data,1);
Kcell = cell(num_data,1);
Qcell = cell(num_data,1);
Fcell = cell(num_data,2);
 

% dimensional reduction

tP = rand(l_d, para.low_dim );
tP = gsch(tP);
tH = tP' * dataY(:, indices == 1); 
tQ = tP;

Pcell{1,1} = tP';
Qcell{1,1} = tQ;
 
%% the first batch
i = 1;
theta = 1./ pdist2(dataX(:,i)', dataX(:, 1:size_ini)');
theta(isinf(theta)) = 0;
theta = sparse(theta);

x_avg = dataX(:,1:size_ini) * theta'/ sum(theta);
y_avg = tH  * theta' / sum(theta);
Ci = dataX(:,1:size_ini)  * diag(theta) * dataX(:,1:size_ini)'  / sum(theta);
Di = tH  * diag(theta) * dataX(:,1:size_ini)' /  sum(theta);
 
A = (Di - y_avg*x_avg') * pinv(Ci - x_avg*x_avg');
b = y_avg - A * x_avg;
tY = A * dataX(:,i) + b;
 
Yj = zeros(l_d, num_data);
Yq = zeros(para.low_dim, num_data);
Yq(:,i) = tY;
Yj(:,i)  = tQ * tY;
Fcell{i,1} = A;
Fcell{i,2} = b;

label_budget = [];

omega = constructW(sparse(dataX'),options);
% omega = zeros(num_data, num_data);
omega(isinf(omega)) = 0;
omega(isnan(omega)) = 0;



 for j = 1:para.num_batch
     
    % A fixed encoder P, and an adaptive decoder tQ
    [j_index, ~] = find(indices == j);
    temp_label_budget = find((label_index == 1) .* (indices == j) == 1);
    label_budget = [label_budget; temp_label_budget];
    tH = tP' * dataY(:, temp_label_budget); 
    tQ = dataY(:, temp_label_budget) * tH' * inv(tH * tH') ;
    tH = tP' * dataY(:, label_budget); 
    
    st = 2 - logical(j - 1); % if j == 1 then st = 2
    for i = st:size(j_index,1)
        id = j_index(i,1);
        [A,b, y_r] = LSR(dataX, Yq, tH, label_budget, id, para, omega);
        Yq(:,id) = y_r;
        Yj(:,id)  = tQ * Yq(:,id);
        Fcell{i,1} = A;
        Fcell{i,2} = b;
    end
    
    % train error 
    train_Y_ground = saveY(:, temp_label_budget);
    [~, train_Y_predicted] = find_k_max_each_column( Yj(:, temp_label_budget), 5);
    Result_train = getResult(train_Y_ground, train_Y_predicted)
    precision_k(sparse(train_Y_predicted),sparse(train_Y_ground),para.tk)
    
    % test errot
    temp_unlabel_budget = find((label_index == 0) .* (indices == j) == 1);
    if isempty(temp_unlabel_budget) == 0
        test_Y_ground = saveY(:, temp_unlabel_budget);
        [~, test_Y_predicted] = find_k_max_each_column( Yj(:, temp_unlabel_budget), 5);
        Result_test = getResult(test_Y_ground, test_Y_predicted)
        precision_k(sparse(train_Y_predicted),sparse(train_Y_ground),para.tk)
    end
    
 end

Yt = saveY(:,unlabel_index);
Yt(Yt == -1) = 0;
Yt(Yt ~= 0) = 1;
% Yc = double(Yj(:,1:size_ini) > 0.1); 

para.tk = 5;
[~,Yc] = find_k_max_each_column(Yj(:,unlabel_index), para.tk);
precision_k(sparse(Yc),sparse(Yt),para.tk)
Result = getResult(Yc,Yt)

% profile viewer