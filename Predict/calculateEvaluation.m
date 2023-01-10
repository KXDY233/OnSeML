function [ac, pr, re, F]=calculateEvaluation(Pre_Labels,test_target)
%
%Pre_Labels: the predicted labels of the classifier, if the ith instance belong to the jth class, Pre_Labels(j,i)=1, otherwise Pre_Labels(j,i)=-1
%test_target: the actual labels of the test instances, if the ith instance belong to the jth class, test_target(j,i)=1, otherwise test_target(j,i)=-1

    [num_class,num_instance]=size(Pre_Labels);
    
    sameC = 0;
    for i=1:size(Pre_Labels,1)
        if(Pre_Labels(i)==1)
            if(Pre_Labels(i)==test_target(i))
                    sameC = sameC + 1;
            end
        end
    end
    allC = sum(Pre_Labels==1) +  sum(test_target==1) - sameC;
    
    ac = sameC/allC;
    if(sum(Pre_Labels==1)>0)
        pr = sameC/sum(Pre_Labels==1);
        re = sameC/sum(test_target==1);
        F = 2*sameC/(sum(Pre_Labels==1) +  sum(test_target==1));
    else
        disp('error..')
    end
    
    
    