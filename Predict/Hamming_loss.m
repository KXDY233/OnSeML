function HammingLoss=Hamming_loss(Pre_Labels,test_target)
%Computing the hamming loss
%Pre_Labels: the predicted labels of the classifier, if the ith instance belong to the jth class, Pre_Labels(j,i)=1, otherwise Pre_Labels(j,i)=-1
%test_target: the actual labels of the test instances, if the ith instance belong to the jth class, test_target(j,i)=1, otherwise test_target(j,i)=-1

    [num_class,num_instance]=size(Pre_Labels);
    miss_pairs=sum(sum(Pre_Labels~=test_target));
	% 内层的sum函数是用于求Pre_Labels与test_target每一列的不一致的个数，输出是一行向量
    HammingLoss=miss_pairs/(num_class*num_instance);