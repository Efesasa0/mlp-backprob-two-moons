function [mse,acc,y]=feedforwardApplication(X,T,V,V0,W,W0)   
    %X:Sxn
    %T:Sxm 
    S=size(X,1);
    V0=repmat(V0,S,1);
    z_in=V0+X*V;%Sxp=Sxp+Sxn.nxp
    z=tanh(z_in);%Sxp
    
    W0=repmat(W0,S,1);
    y_in=W0+z*W;%Sxm=Sxm+Sxp.pxm
    y=tanh(y_in);%Sxm

    mse=mean((T-y).^2);%Sxm-Sxm %MSE calculate
    acc=sum(sign(y)==T)/S; %Accuracy calculate
end