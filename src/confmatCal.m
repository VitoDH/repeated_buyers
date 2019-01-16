function [] = confmatCal(confmat)

    TP = confmat(1,1);

    FN = confmat(1,2);

    FP = confmat(2,1);

    TN = confmat(2,2);

    N = TN + FP;

    P = TP + FN;


    fprintf('%s %d\n','TP =', TP);

    fprintf('%s %d\n','TN =', TN);

    fprintf('%s %d\n','FP =', FP);

    fprintf('%s %d\n\n','FN =', FN);

    fprintf('%s %d\n','N =', N);

    fprintf('%s %d\n\n','P =', P);

    

    fprintf('%s %d\n','TP_Rate =', TP/N);

    fprintf('%s %d\n','FP_Rate =', FP/N);

    fprintf('%s %d\n','Specificity =', TN/N);

    fprintf('%s %d\n','Recall =', TP/P);

    fprintf('%s %d\n','Precision =', TP/(TP+FP) );

    fprintf('%s %d\n\n','Accuracy =', (TP+TN)/(P+N) );


    fprintf('%s %d\n','g_mean1 =', sqrt(TP*P) );

    fprintf('%s %d\n','g_mean2 =', sqrt(TP*TN) );

end