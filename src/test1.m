%user_log = csvread('user_log_format1.csv',1);
totalSet = xlsread('trainSet2.xls');
format bank;
%pΪ���룬tΪ���
p = totalSet(:,3:12)';
t = totalSet(:,13)';
%���ݵĹ�һ������������mapminmax������ʹ��ֵ��һ����[-1.1]֮��
%�ú���ʹ�÷������£�[y,ps] =mapminmax(x,ymin,ymax)��x��黯���������룬
%ymin��ymaxΪ��黯���ķ�Χ������Ĭ��Ϊ�黯��[-1,1]
%���ع黯���ֵy���Լ�����ps��ps�ڽ������һ���У���Ҫ����
[p1,ps]=mapminmax(p);
[t1,ts]=mapminmax(t);
%ȷ��ѵ�����ݣ���������,һ��������Ĵ�������ѡȡ70%��������Ϊѵ������
%15%��������Ϊ�������ݣ�һ����ʹ�ú���dividerand����һ���ʹ�÷������£�
%[trainInd,valInd,testInd] = dividerand(Q,trainRatio,valRatio,testRatio)
[trainsample.p,valsample.p,testsample.p] =dividerand(p1,0.7,0.15,0.15);
[trainsample.t,valsample.t,testsample.t] =dividerand(t1,0.7,0.15,0.15);
%�������򴫲��㷨��BP�����磬ʹ��newff��������һ���ʹ�÷�������
%net = newff(minmax(p),[�������Ԫ�ĸ�������������Ԫ�ĸ���],{������Ԫ�Ĵ��亯���������Ĵ��亯����,'���򴫲���ѵ������'),����pΪ�������ݣ�tΪ�������
%tfΪ������Ĵ��亯����Ĭ��Ϊ'tansig'����Ϊ����Ĵ��亯����
%purelin����Ϊ�����Ĵ��亯��
%һ�������ﻹ�������Ĵ���ĺ���һ������£����Ԥ�������Ч�����Ǻܺã����Ե���
%TF1 = 'tansig';TF2 = 'logsig';
%TF1 = 'logsig';TF2 = 'purelin';
%TF1 = 'logsig';TF2 = 'logsig';
%TF1 = 'purelin';TF2 = 'purelin';
TF1='tansig';TF2='tansig';TF3='purelin';
net=newff(minmax(p1),[100,50,1],{TF1 TF2 TF3},'trainscg');%���紴��
%�������������
net.trainParam.epochs=10000;%ѵ����������
net.trainParam.goal=1e-7;%ѵ��Ŀ������
net.trainParam.lr=0.01;%ѧϰ������,Ӧ����Ϊ����ֵ��̫����Ȼ���ڿ�ʼ�ӿ������ٶȣ����ٽ���ѵ�ʱ�����������������ʹ�޷�����
net.trainParam.mc=0.9;%�������ӵ����ã�Ĭ��Ϊ0.9
net.trainParam.show=25;%��ʾ�ļ������
% ָ��ѵ������
% net.trainFcn = 'traingd'; % �ݶ��½��㷨
% net.trainFcn = 'traingdm'; % �����ݶ��½��㷨
% net.trainFcn = 'traingda'; % ��ѧϰ���ݶ��½��㷨
% net.trainFcn = 'traingdx'; % ��ѧϰ�ʶ����ݶ��½��㷨
% (�����������ѡ�㷨)
% net.trainFcn = 'trainrp'; % RPROP(����BP)�㷨,�ڴ�������С
% �����ݶ��㷨
% net.trainFcn = 'traincgf'; %Fletcher-Reeves�����㷨
% net.trainFcn = 'traincgp'; %Polak-Ribiere�����㷨,�ڴ������Fletcher-Reeves�����㷨�Դ�
% net.trainFcn = 'traincgb'; % Powell-Beal��λ�㷨,�ڴ������Polak-Ribiere�����㷨�Դ�
% (�����������ѡ�㷨)
%net.trainFcn = 'trainscg'; % ScaledConjugate Gradient�㷨,�ڴ�������Fletcher-Reeves�����㷨��ͬ,�����������������㷨��С�ܶ�
% net.trainFcn = 'trainbfg'; %Quasi-Newton Algorithms - BFGS Algorithm,���������ڴ�������ȹ����ݶ��㷨��,�������ȽϿ�
% net.trainFcn = 'trainoss'; % OneStep Secant Algorithm,���������ڴ��������BFGS�㷨С,�ȹ����ݶ��㷨�Դ�
% (�����������ѡ�㷨)
%net.trainFcn = 'trainlm'; %Levenberg-Marquardt�㷨,�ڴ��������,�����ٶ����
% net.trainFcn = 'trainbr'; % ��Ҷ˹�����㷨
% �д����Ե������㷨Ϊ:'traingdx','trainrp','trainscg','trainoss', 'trainlm'
%������һ����ѡȡ'trainlm'������ѵ��������Զ�Ӧ����Levenberg-Marquardt�㷨
net.trainFcn='trainscg';
[net,tr]=train(net,trainsample.p,trainsample.t);
%������棬��һ����sim����
[normtrainoutput,trainPerf]=sim(net,trainsample.p,[],[],trainsample.t);%ѵ�������ݣ�����BP�õ��Ľ��
[normvalidateoutput,validatePerf]=sim(net,valsample.p,[],[],valsample.t);%��֤�����ݣ���BP�õ��Ľ��
[normtestoutput,testPerf]=sim(net,testsample.p,[],[],testsample.t);%�������ݣ���BP�õ��Ľ��
%�����õĽ�����з���һ�����õ�����ϵ�����,Ԥ���ǩ����һ��
trainoutput=mapminmax('reverse',normtrainoutput,ts);
validateoutput=mapminmax('reverse',normvalidateoutput,ts);
testoutput=mapminmax('reverse',normtestoutput,ts);
%������������ݵķ���һ���Ĵ������õ�����ʽֵ��ʵ��ֵ����һ��
trainvalue=mapminmax('reverse',trainsample.t,ts);%��������֤����
validatevalue=mapminmax('reverse',valsample.t,ts);%��������֤������
testvalue=mapminmax('reverse',testsample.t,ts);%�����Ĳ�������
%��Ԥ�⣬����ҪԤ�������pnew
%pnew=[313,256,239]';
%pnewn=mapminmax(pnew);
%anewn=sim(net,pnewn);
%anew=mapminmax('reverse',anewn,ts);
%�������ļ���
errors=trainvalue-trainoutput;
%plotregression���ͼ
figure,plotregression(trainvalue,trainoutput)
%���ͼ
figure,plot(1:length(errors),errors,'-b')
title('���仯ͼ')
%���ֵ����̬�Եļ���
figure,hist(errors);%Ƶ��ֱ��ͼ
figure,normplot(errors);%Q-Qͼ
[muhat,sigmahat,muci,sigmaci]=normfit(errors); %�������� ��ֵ,����,��ֵ��0.95��������,�����0.95��������
[h1,sig,ci]= ttest(errors,muhat);%�������
figure, ploterrcorr(errors);%�������������ͼ
figure, parcorr(errors);%����ƫ���ͼ

%ѵ�������Ļ�������
trainConf = confusionmat(trainvalue,double(trainoutput>0.07));
trainAUC=roc_curve(trainoutput',trainvalue');
%��֤�����Ļ�������
validateConf = confusionmat(validatevalue,double(validateoutput>0.07));
validateAUC=roc_curve(validateoutput',validatevalue');
%���������Ļ�������
testConf = confusionmat(testvalue,double(testoutput>0.07));
testAUC=roc_curve(testoutput',testvalue');









%{

trainX = totalSet(1:9000,3:12)';
trainX = mapminmax(trainX);
trainY =totalSet(1:9000,13)';

testX = totalSet(9001:9854,3:12)';
testX = mapminmax(testX);
testY =totalSet(9001:9854,13)';

predicttrain = network1_outputs';
predtrain = double((predicttrain>0.1)');
trainConf = confusionmat(trainY,predtrain);

predicttest = network1_outputstest';
predtest = double((predicttest>0.1)');
testConf = confusionmat(testY,predtest);

%}