function [A,b,y_r] = LSR(dataX, Yq, tH, label_budget, id, para, omega)   

theta = 1./ pdist2(dataX(:,id)', dataX(:, label_budget)');
theta(isinf(theta)) = 0;

t3 = sum(omega(1:(id-1), id));
t1 = sum(theta) + para.lambda * t3;

x_avg = ( dataX(:, label_budget) * theta' +...
    para.lambda * t3 * dataX(:,id) ) / t1;

t2 = omega(1:(id-1), id);
t2(size(dataX,2),1) = 0;
y_avg = ( tH * theta' + para.lambda * Yq * t2 ) / t1;

Ci = ( dataX(:, label_budget) * diag(sparse(theta)) * dataX(:, label_budget)' + ...
       para.lambda *  sum(t2) * dataX(:,id) *dataX(:,id)') / t1;

 % sparse diag 3.335s
 % diag 4.8s
 % diag sparse  2.23s
   
Di = ( tH * diag(sparse(theta)) * dataX(:, label_budget)' + ...
       para.lambda * Yq * t2 * dataX(:,id)' ) / t1;
cc = Ci - x_avg*x_avg' + eye(size(Ci,1));
% [~,U] = lu(cc);
% tU = invutri(U);
% iU = tU * tU';
% A = (Di - y_avg*x_avg') * iU;
A = (Di - y_avg*x_avg') / cc;
b = y_avg - A*x_avg;
y_r = A * dataX(:, id) + b;

end