clc;
clearvars;
X=rand(100,1);

Y = cos(2*pi*X)+tanh(2*pi*X);
T=[];
    T=cos(2*pi*X)+tanh(2*pi*X)+normrnd(0,0.1,[100,1]);
  
M = 3;
trainDS = 70;
Xtrain = X(1:trainDS);
Ttrain = T(1:trainDS);
Ytrain = Y(1:trainDS);
[Xtrain,I] = sort(Xtrain);
Ytrain = Ytrain(I,:);
Ttrain = Ttrain(I,:);

figure(1);
scatter(Xtrain,Ttrain);
hold on;
fplot(@(X) cos(2*pi*X)+tanh(2*pi*X),[0 1],'r');
hold on;

D = zeros(trainDS,M+1);

for r = 1:trainDS
    for c = 1:M+1
        D(r,c) = Xtrain(r)^(c-1);
    end
end

L = 0.0000000000000001;
WI = inv((D'*D) + L * eye(M+1)) * D';
%WI = pinv(D) + L*D';
W = WI * Ttrain;

y = D * W;
phelp = fliplr(W');
%phelp = fliplr(phelp);
shelp = poly2sym(phelp);
fplot(shelp);
%yl = min(min(y),min(Ttrain));
%yh = max(max(y),max(Ttrain));;
axis([0 1 -2 2]);
%plot(Xtrain,polyval(W,Xtrain));
%fplot(@(x) poly2sym(sym(W)),[0 1],'g');
%plot(Xtrain,y,'color','green','LineWidth',2);
legend('with noise','original','approximated');

title('Training Data');
xlabel('xn');
ylabel('yn');
hold off;
Etrain = error(y,Ttrain,L,W);

figure(2);
dim = [.6 .4 .4 .5];
scatter(Ttrain,y);
axis([0 2 0 2]);
str  = { strcat('M= ',num2str(M)),strcat('L= ',num2str(L))};
annotation('textbox',dim,'String',str,'FitBoxToText','on');
title('Training Data With Best Case');
xlabel('tn');
ylabel('y(x,w)');
Xval = X(71:80);
Tval = T(71:80);
Yval = Y(71:80);

Dval = ones(10,M+1);
for r = 1:10
    for c = 1:M+1
        Dval(r,c) = Xval(r)^(c-1);
    end
end
y1 = Dval * W;
Eval = error(y1,Tval,L,W); 

Xtest = X(81:100);
Ttest = T(81:100);
Ytest = Y(81:100);
Dtest = ones(20,M+1);
for r = 1:20
    for c = 1:M+1
        Dtest(r,c) = Xtest(r)^(c-1);
    end
end

 y2 = Dtest * W;
 
figure(3);
dim = [.7 .4 .4 .5];
scatter(Ttest,y2);
axis([0 2 0 2]);
str  = { strcat('M= ',num2str(M)),strcat('L= ',num2str(L))};
title('Test Data With Best Case');
xlabel('tn');
ylabel('y(x,w)');
annotation('textbox',dim,'String',str,'FitBoxToText','on');
Etest = error(y2,Ttest,L,W);
 
 function rms = error(y,t,L,W)
    rms = 0;
    for c = 1:length(t)
        rms = rms+(y(c)-t(c))^2;
    end
    rms = rms / length(t);
    rms = rms^(1/2);
    rms = rms + L/2 * (W'*W);
end