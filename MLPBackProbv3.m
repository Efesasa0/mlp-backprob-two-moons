clear variables
close all
clc
rng(21)
% Please set setTest to either 1 or 2.
% setTest=1 means Train Set1, Test on Set2
% setTest=2 means Train Set2, Test on Set1
setTest=2;
%------------------
% Get Data
%------------------

if(setTest==1)
    % Train Set1, Test on Set2
    dataSetTrain='DataSet1_MP1.mat';
    dataSetTest='DataSet2_MP1.mat';

    Struct1=load(dataSetTrain); % Training set
    Xall=Struct1.DataSet1;
    Tall=Struct1.DataSet1_targets;
    
    Struct2test=load(dataSetTest); % Testing set
    Xtest=Struct2test.DataSet2;
    Ttest=Struct2test.DataSet2_targets;

elseif(setTest==2)    
    % or train Set2, Test on Set1
    dataSetTrain='DataSet2_MP1.mat';
    dataSetTest='DataSet1_MP1.mat';
    
    Struct1=load(dataSetTrain); % Training set
    Xall=Struct1.DataSet2;
    Tall=Struct1.DataSet2_targets;
    
    Struct2test=load(dataSetTest); % Testing set
    Xtest=Struct2test.DataSet1;
    Ttest=Struct2test.DataSet1_targets;
end

%-----------------------
% Shuffle, Seperate Data
%-----------------------
Sall=size(Xall,1);
idx=randperm(Sall);
cutpoint=floor(0.8*size(idx,2));
trainidx=idx(1:cutpoint);
valididx=idx(cutpoint+1:end);

Xtrain=Xall(trainidx,:);
Xvalid=Xall(valididx,:);
Ttrain=Tall(trainidx,:);
Tvalid=Tall(valididx,:);

X=Xtrain;
T=Ttrain;

%------------------
% MLP NN Structure
%------------------
S=size(X,1);
n=size(X,2);%2
p=20;%20
m=size(T,2);%1
maxEpoch=30;%30

%------------------
% Weight Initialization
%------------------
%between 0.5 and -0.5
% V0=(-0.5-0.5).*rand(1,p) + 0.5;%1xp
% W0=(-0.5-0.5).*rand(1,m) + 0.5;%1xm
% V=(-0.5-0.5).*rand(n,p) + 0.5;%nxp
% W=(-0.5-0.5).*rand(p,m) + 0.5;%pxm

%between 1 and -1
% V0=-0.1+(0.2).*rand(1,p);%1xp
% W0=-0.1+(0.2).*rand(1,m);%1xm
% V=-0.1+(0.2).*rand(n,p);%nxp
% W=-0.1+(0.2).*rand(p,m);%pxm

%Nguyen-Widrow
B=0.7*(p)^(1/n);
Vold=(-0.5-0.5).*rand(n,p) + 0.5;%1xp
Vnorm=sqrt(sum(Vold.^2));
V=(B*Vold)./Vnorm;
V0=((-B)-B).*rand(1,p) + B;%1xp
W=(-0.5-0.5).*rand(p,m) + 0.5;%pxm
W0=(-0.5-0.5).*rand(1,m) + 0.5;%1xm

%------------------
% Train the NN
%------------------
alphaStart=1e-1;
alphaEnd=1e-5;

W1t=zeros(p,m);%%%%%%%
W2t=zeros(p,m);%%%%%%%
V1t=zeros(n,p);%%%%%%%
V2t=zeros(n,p);%%%%%%%

TrainError=zeros(maxEpoch,1);
TrainAcc=zeros(maxEpoch,1);
ValidError=zeros(maxEpoch,1);
ValidAcc=zeros(maxEpoch,1);


for epoch=1:maxEpoch
    %Alpha calculate
    a=(alphaEnd-alphaStart)/(maxEpoch-1);
    b=alphaStart-a;
    alpha=a*epoch+b;

    momentum=a*epoch+b;%%%%%%%
    
    fprintf('epoch: %d, alpha: %f, ',epoch,alpha)

    for s=1:S
        x=X(s,:);%1xn
        t=T(s,:);%1xm

        z_in=V0+x*V;%1xp=1xp+1xn.nxp
        z=tanh(z_in);%1xp

        y_in=W0+z*W;%1xm=1xm+1xp.pxm
        y=tanh(y_in);%1xm

        %-------------
        % Backprob
        %-------------

        deltak=(t-y).*tanhD(y_in);%1xm=(1xm-1xm).1xm
        deltaW=alpha*z'*deltak;%pxm=1x1.px1.1xm
        deltaW0=alpha*deltak;%1xm=1x1.1xm
        
        delta_inj=deltak*W';%1xp=1xm.mxp
        
        deltaj=delta_inj.*tanhD(z_in);%1xp=1xp.1xp
        deltaV=alpha*x'*deltaj;%nxp=1x1.1xn.1xp
        deltaV0=alpha*deltaj;%1xp=1x1.1xp

        W=W1t+deltaW+momentum*(W1t-W2t);%pxm=pxm+pxm+1x1*(pxm-pxm)
        W0=W0+deltaW0;%1xm=1xm+1xm
        V=V1t+deltaV+momentum*(V1t-V2t);%nxp=nxp+nxp+1x1*(nxm-nxm)
        V0=V0+deltaV0;%1xp=1xp.1xp
        if s>=1
            if s>=2
                W2t=W1t;
                V2t=V1t;
            end
            W1t=W;
            V1t=V;
        end
    end
    %Training Check
    [trainMse,trainAcc]=feedforwardApplication(X,T,V,V0,W,W0);
    [validMse,validAcc]=feedforwardApplication(Xvalid,Tvalid,V,V0,W,W0);
    fprintf('Train_MSE: %f, Train_Acc: %f, Valid_MSE: %f, Valid_Acc: %f\n',trainMse, trainAcc, validMse,validAcc)
    %Train and Valid Store
    TrainError(epoch)=trainMse;
    TrainAcc(epoch)=trainAcc;
    ValidError(epoch)=validMse;
    ValidAcc(epoch)=validAcc;
    if trainAcc==1
        break
    end
    
end

%Plot MSE of Training and Validation

plot(1:epoch,TrainError(1:epoch),LineWidth=2)
hold on
plot(1:epoch,ValidError(1:epoch),LineWidth=2)
legend('TrainError','ValidError');
title(sprintf('Training and Validation MSEs with maxEpoch=%d\n on data %s',maxEpoch,dataSetTrain))

figure
plot(1:epoch,TrainAcc(1:epoch),LineWidth=2)
hold on
plot(1:epoch,ValidAcc(1:epoch),LineWidth=2)
legend('TrainAcc','ValidAcc');
title(sprintf('Training and Validation Accuracies with maxEpoch=%d\n on data %s',maxEpoch,dataSetTrain))

%Final Test on dataSetTest
[testMse,testAcc,ytest]=feedforwardApplication(Xtest,Ttest,V,V0,W,W0);
fprintf('Test on %s, MSE:%f, Acc:%f\n',dataSetTest,testMse,testAcc);

figure
x1range = -15:0.25:25;
x2range = -15:0.25:15;
[xx1, xx2] = meshgrid(x1range,x2range);
Xgrid = [xx1(:) xx2(:)];
[~,~,prediction]=feedforwardApplication(Xgrid,zeros(size(Xgrid,1),1),V,V0,W,W0);
prediction=sign(prediction);
gscatter(xx1(:), xx2(:), prediction,'rgb');
legend off, axis tight
title(sprintf('Testing decision boundry of %s',dataSetTest))
hold on
ytest=sign(ytest);
scatter(Xtest(ytest==1,1),Xtest(ytest==1,2))
scatter(Xtest(ytest==-1,1),Xtest(ytest==-1,2))


