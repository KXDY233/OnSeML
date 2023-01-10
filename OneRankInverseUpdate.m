% % % A function for rank-1 update for the Moore-Penrose pseudo-inverse of real valued matrices
% % % The Matrix Cookbook [ http://matrixcookbook.com ]
% % % 3.2.7 Rank-1 update of Moore-Penrose Inverse
% % % When the matrix A is updated by the product of two vectors c,d: A+cd'
% % % The inputs are :  A, Original Matrix
% % %                   A_pinv, Pseudo Inverse of Matrix A 
% % %                   c, the first input vector
% % %                   d, the second input vector
% % % and the last two optional inputs are: 
% % % A tolerance for checking zero conditions, to conver numerical errors;
% % % A 0/1 flag to print or not print the condition of two input  vectors 
% % % with respect to the matrix A, by setting it to 1 and 0, respectively.
%% 
function A_pinv_New = OneRankInverseUpdate(A,A_pinv,c,d,Zero_tol,Case_Print_Flag)
%% Input Checks 
% check the number of inputs
narginchk(4, 6)
% check the dimension of inputs
[n,m] = size(A);
if ~isequal(size(A_pinv),[m,n])
    error('The input pseudo inverse matrix dimension mismatch');
end
if ~isequal(size(c),[n 1])
    error('The first input vector dimension mismatch');
end
if ~isequal(size(d),[m 1])
    error('The second input vector dimension mismatch');
end
% set a small value instead of zero, for avoiding numverical issues
if ~exist('Zero_tol','var')
    Zero_tol = ((eps));
else 
    if Zero_tol>=1 || Zero_tol<=0
        error('The zero tolerance value should be a very small positive value, and must be between 0 and 1');
    end
end
% set the case number printing status
if ~exist('Case_Print_Flag','var')
    Case_Print_Flag = 1;
else 
    if ~ismember(Case_Print_Flag,[0,1]) 
        error('The last input can only be 0 or 1, i.e. print or hide case number, respectively');
    end
end
%% Initial Calculations
V = A_pinv*c;
b = 1 + d'*V;
N = A_pinv'*d;
W = (eye(n) - A*A_pinv)*c;
M = (eye(m) - A_pinv*A)*d;
% squared norm of the two abovesaid vectors
w_snorm = norm(W,2)^2;  
m_snorm = norm(M,2)^2;
%% Computation of the update term 
if w_snorm>=Zero_tol && m_snorm>=Zero_tol
    if Case_Print_Flag == 1
        disp('case 1');
    end
    G = (-1/w_snorm) * V * W' - (1/m_snorm) * M * N' + (b/m_snorm/w_snorm) * M * W';
elseif w_snorm<Zero_tol && m_snorm>=Zero_tol && abs(b)<Zero_tol
    if Case_Print_Flag == 1
        disp('case 2'); 
    end    
    v_snorm = norm(V,2)^2;    
    G = (-1/v_snorm) * (V * V') * A_pinv -(1/m_snorm) * M * N';
elseif w_snorm<Zero_tol && abs(b)>Zero_tol
    if Case_Print_Flag == 1
        disp('case 3');
    end    
    v_snorm = norm(V,2)^2;    
    G = (1/b) * M * V' * A_pinv - (b/(v_snorm * m_snorm + b^2)) *...
        ((v_snorm / b) * M + V) * ((m_snorm / b) * A_pinv' * V + N)';        
elseif m_snorm<Zero_tol && w_snorm>=Zero_tol && abs(b)<Zero_tol
    if Case_Print_Flag == 1
        disp('case 4');
    end    
    n_snorm = norm(N,2)^2;
    G = (-1/n_snorm) * A_pinv * (N * N') -(1/ w_snorm) * V * W';
elseif m_snorm<Zero_tol && abs(b)>Zero_tol
    if Case_Print_Flag == 1
        disp('case 5');        
    end    
    n_snorm = norm(N,2)^2;    
    G = (1/b) * A_pinv * N * W' - (b/(n_snorm * w_snorm + b^2)) *...
        ((w_snorm/b) * A_pinv * N + V) * ((n_snorm/b) * W + N)';
elseif m_snorm<Zero_tol && w_snorm<Zero_tol && abs(b)<Zero_tol
    if Case_Print_Flag == 1
        disp('case 6');
    end    
    v_snorm = norm(V,2)^2;
    n_snorm = norm(N,2)^2;
    G = (-1/v_snorm) * (V * V') * A_pinv - (1/n_snorm) * A_pinv * (N * N') + ...
        V' * A_pinv * N / (v_snorm * n_snorm) * (V * N');
end
%% Computation of the New Pseudo Inverse
A_pinv_New = A_pinv + G;
end