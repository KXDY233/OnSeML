clear all; close all; clc;
addpath(genpath('./Tools'));
addpath('./Predict');

profile on

%% Data processing

para.filename = 'rcv1s1'; %Corel5k, bibtex delicious mediamill
[dataX, dataY] = load_data(para.filename);

% remove data without any labels
nll = sum(dataY, 1);
inx = find(nll ~= 0);
dataX = dataX(:, inx);
dataY = dataY(:, inx);

saveX = dataX;
saveY = dataY;

dataY = full(dataY);
dataY(dataY == 0) = -1;

f_d = size(dataX,1); % fea_dimension
l_d = size(dataY,1); % label_dimension
num_data = size(dataX,2);

para.labelled = 0.1; % Other partly labelled data. x*size(data)
para.low_dim = 15;
para.lambda = 0.001;
para.blsize = 50000;
para.basize = 50000;
para.qsize = 5;

para.tk = 5; % number of predicted labels

options = [];
options.WeightMode = 'HeatKernel'; % Cosine HeatKernel
options.bSelfConnected = 1;
options.k = 10;   

size_ini  = 2; %round(num_data * para.flabelled);
num_unlabelled = num_data - round(num_data * para.labelled-size_ini);

unlabel_index = size_ini + randperm(num_data - size_ini, num_unlabelled);
dataY(:,unlabel_index) = -1;

% label_index place a flag of labelled/unlabelled
label_index = ones(num_data, 1);
label_index(unlabel_index) = 0;

[label_bu,~] = find(label_index == 1);

% dimensional reduction
tP = rand(l_d, para.low_dim);
tP = gsch(tP);

tQ = tP;
tH = tP' * dataY(:, 1:size_ini);

%% for the first data
i = 1;
theta = 1./ pdist2(dataX(:,i)', dataX(:, 1:size_ini)');
theta(isinf(theta)) = 0;

x_avg = dataX(:,1:size_ini) * theta'/ sum(theta);
y_avg = tH  * theta' / sum(theta);
Ci = dataX(:,1:size_ini)  * diag(theta) * dataX(:,1:size_ini)'  / sum(theta);
Di = tH  * diag(theta) * dataX(:,1:size_ini)' /  sum(theta);

A = (Di - y_avg*x_avg') * pinv(Ci - x_avg*x_avg' + 0.001 * eye(size(Ci)));
b = y_avg - A * x_avg;
tY = A * dataX(:,i) + b;

Yj = zeros(l_d, num_data);
Yq = zeros(para.low_dim, num_data);
Yq(:,i) = tY;
Yj(:,i)  = tQ * tY;
Fcell{i,1} = A;
Fcell{i,2} = b;

label_budget = [];

omega = constructW(dataX',options);
omega(isinf(omega)) = 0;
omega(isnan(omega)) = 0;

label_budget = 1:2;
temp_label_budget = [];
count = 0;

tH = tP' * dataY(:, label_budget); 
tQ = tP ;

BL = 1:1;
BA = 1:1;

for i = 2:num_data 
    
    tH = tP' * dataY(:, BL); 
    theta = 1./ pdist2(dataX(:,i)', dataX(:, BL)');
    theta(isinf(theta)) = 0;
    
    t3 = sum(omega(BA, i));
    t1 = sum(theta) + para.lambda * t3;
    
    x_avg = ( dataX(:, BL) * theta' +...
        para.lambda * t3 * dataX(:,i) ) / t1;
    
    t2 = omega(BA, i);
    t2(size(dataX,2),1) = 0;
    
    y_avg = ( tH * theta' + para.lambda * Yq * t2 ) / t1;
    
    Ci = ( dataX(:, BL) * diag(sparse(theta)) * dataX(:, BL)' + ...
        para.lambda *  sum(t2) * dataX(:,i) *dataX(:,i)') / t1;
    
    Di = ( tH * diag(sparse(theta)) * dataX(:, BL)' + ...
        para.lambda * Yq * t2 * dataX(:,i)' ) / t1;
    
    A = (Di - y_avg*x_avg') * pinv(Ci - x_avg*x_avg' + 0.001* eye(size(Ci,1)));
    b = y_avg - A * x_avg;
    tY = A * dataX(:,i) + b;
    Yq(:,i) = tY;
    Yj(:,i)  = tQ * tY;
    
    if label_index(i) == 1
        BL = [BL, i];
        BA = [BA, i];
    else
        BA = [BA, i];
    end
    
    if size(BA,2) >= para.basize
        BA(:, 1) = [];
    end
    if size(BL,2) >= para.blsize
        BL(:, 1) = [];
    end 
    
    if i > 100 && label_index(i) == 1
        count = count + 1;
        temp_label_budget = [temp_label_budget, i];
        if count == para.qsize
            label_budget = [label_budget, temp_label_budget];
            tH = tP' * dataY(:, label_budget); 
            tQ = dataY(:, label_budget) * tH' * inv(tH * tH');
            count = 0;
            temp_label_budget = [];
        end    
    end
end
    
Yt = saveY(:,unlabel_index);
Yt(Yt == -1) = 0;
Yt(Yt ~= 0) = 1;
[~,Yc] = find_k_max_each_column(Yj(:,unlabel_index), para.tk);


precision_k(sparse(Yc),sparse(Yt),5)
Result = getResult(Yc,Yt)
