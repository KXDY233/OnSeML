function fscore = FScore(Y_hatM,YM)
%FScore: Compute the FScore
%
%          Pred          L x Nt predicted label matrix           
%          Yt            L x Nt groundtruth label matrix

numNt  = size(YM,2);
F1cnt=0;
count  = numNt;
for i = 1 : numNt
    Y_hat=Y_hatM(:,i);
    Y=YM(:,i);
    Y(Y<=0) = -1;
    Y(Y>0) = 1;
    Y_hat(Y_hat<=0) = -1;
    Y_hat(Y_hat>0) = 1;
    
    % start
    n = size(Y,1);
    TP = nnz(Y_hat==1&Y==1);
    FP = nnz(Y_hat==1&Y==-1);
    FN = nnz(Y_hat==-1&Y==1);
    TN = nnz(Y_hat==-1&Y==-1);
    
    if n~=(TP+FP+FN+TN)
        disp('n~=(TP+FP+FN+TN)!');
        F1 = 0;
    else
        if TP+FP == 0
            prec = 0;
        else
            prec = TP/(TP+FP);
        end
        if TP + FN == 0
            rec = 0.5;
        else
            rec = TP/(TP+FN);
        end
        if prec+rec == 0
            F1 = 0;
        else
            F1 = 2*prec*rec / (prec+rec);
        end
        F1cnt=F1cnt+F1;
end
fscore = F1cnt / count;

end