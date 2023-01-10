function [numDiff, numSame, numPart] = nRatio(M)

numDiff = size(M(M == 1),1)
numSame = size(M(M == 0),1)
numPart = size(M,1) * size(M,2) - numDiff - numSame