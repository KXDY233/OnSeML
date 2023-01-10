clear all; close all; clc;
addpath(genpath('./Tools'));
addpath('./Predict');

profile on

%% Data processing

para.filename = 'mirflickr'; % bibtex delicious
para.labelled = 0.1; % Other partly labelled data. x*size(data)
para.low_dim = 6;
para.lambda = 0.001;
para.blsize = 50000;
para.basize = 50000;
para.qsize = 5;

para.tk = 5; % number of predicted labels

options = [];
options.WeightMode = 'HeatKernel'; % Cosine HeatKernel
options.bSelfConnected = 1;
options.k = 20;     

precision = {};
Result = {};

%parpool('local', 5);

for it = 1:5
    
    [dataX, dataY] = load_data(para.filename);
    nll = sum(dataY, 1);
    inx = find(nll ~= 0);
    dataX = dataX(:, inx);
    dataY = dataY(:, inx);
    dataX = dataX(:, 1:10000);
    dataY = dataY(:, 1:10000);
    saveX = dataX;
    saveY = dataY;
    
    dataY = full(dataY);
    dataY(dataY == 0) = -1;
    
    f_d = size(dataX,1); % fea_dimension
    l_d = size(dataY,1); % label_dimension
    num_data = size(dataX,2);
    
    size_ini  = 2; %round(num_data * para.flabelled);
    num_unlabelled = num_data - round(num_data * para.labelled-size_ini);
    
    unlabel_index = size_ini + randperm(num_data - size_ini, num_unlabelled);
    dataY(:,unlabel_index) = -1;
    
    % label_index place a flag of labelled/unlabelled
    label_index = ones(num_data, 1);
    label_index(unlabel_index) = 0;
    
    [label_bu,~] = find(label_index == 1);
    
    
    %% Initilization random label compression
    % generate an orthogonalized encoding matrix P_0
    % get the related decoding matrix Q_0 (with H_0 and K_0)
    
    
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
    
    %     A = (Di - y_avg*x_avg') * pinv(Ci - x_avg*x_avg' + 0.01 * eye(size(Ci)));
    A = (Di - y_avg*x_avg') * pinv(Ci - x_avg*x_avg' );
    b = y_avg - A * x_avg;
    tY = A * dataX(:,i) + b;
    
    Yj = zeros(l_d, num_data);
    Yq = zeros(para.low_dim, num_data);
    Yq(:,i) = tY;
    Yj(:,i)  = tQ * tY;
    
    label_budget = [];
    
    omega = constructW(dataX',options);
    omega(isinf(omega)) = 0;
    omega(isnan(omega)) = 0;
    
    label_budget = 1:2;
    temp_label_budget = [];
    count = 0;
    
    tH = tP' * dataY(:, label_budget);
    tQ = tP;
    %tQ = dataY(:, label_budget) * tH' / (tH * tH') ;
    
    for i = 2:num_data
        
        tH = tP' * dataY(:, label_budget);
        theta = 1./ pdist2(dataX(:,i)', dataX(:, label_budget)');
        theta(isinf(theta)) = 0;
        
        t3 = sum(omega(1:(i-1), i));
        t1 = sum(theta) + para.lambda * t3;
        
        x_avg = ( dataX(:, label_budget) * theta' +...
            para.lambda * t3 * dataX(:,i) ) / t1;
        
        t2 = omega(1:(i-1), i);
        t2(size(dataX,2),1) = 0;
        
        y_avg = ( tH * theta' + para.lambda * Yq * t2 ) / t1;
        
        Ci = ( dataX(:, label_budget) * diag(sparse(theta)) * dataX(:, label_budget)' + ...
            para.lambda *  sum(t2) * dataX(:,i) *dataX(:,i)') / t1;
        
        Di = ( tH * diag(sparse(theta)) * dataX(:, label_budget)' + ...
            para.lambda * Yq * t2 * dataX(:,i)' ) / t1;
        
        A = (Di - y_avg*x_avg') * pinv(Ci - x_avg*x_avg' + 0.001 * eye(size(Ci,1)));
        b = y_avg - A * x_avg;
        tY = A * dataX(:,i) + b;
        Yq(:,i) = tY;
        Yj(:,i)  = tQ * tY;
        
        
        if i > para.low_dim && label_index(i) == 1
            count = count + 1;
            temp_label_budget = [temp_label_budget, i];
            if count == para.qsize
                label_budget = [label_budget, temp_label_budget];
                tH = tP' * dataY(:, label_budget);
                tQ = dataY(:, label_budget) * tH' * pinv(tH * tH');
                count = 0;
                temp_label_budget = [];
            end
        end
    end
    
    Yt = saveY(:,unlabel_index);
    Yt(Yt == -1) = 0;
    Yt(Yt ~= 0) = 1;
    [~,Yc] = find_k_max_each_column(Yj(:,unlabel_index), 5);
    precision{it} = precision_k(sparse(Yj(:,unlabel_index)),sparse(Yt),5);
    Result{it} = getResult(Yc,Yt);
    
    precision_k(sparse(Yj(:,unlabel_index)),sparse(Yt),5)
    getResult(Yc,Yt)
    
end

prepre = cell2mat(precision);
ave_pre = mean(prepre');
std_pre = std(prepre');

disp( [mat2str(ave_pre(1,1),4), '\pm', mat2str(std_pre(1,1),3)]);
disp( [mat2str(ave_pre(1,3),4), '\pm', mat2str(std_pre(1,3),3)]);
disp( [mat2str(ave_pre(1,5),4), '\pm', mat2str(std_pre(1,1),5)]);

resres = cell2mat(Result);
resres = reshape(resres, 7, 5);
ave_res = mean(resres');
std_res = std(resres');

disp( [mat2str(ave_res(1,4),4), '\pm', mat2str(std_res(1,4),3)]);
disp( [mat2str(ave_res(1,7),4), '\pm', mat2str(std_res(1,7),3)]);

