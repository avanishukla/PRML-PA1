clc;
clearvars;
mydir = '/home/neha/workspace/data_assign1_group12/';
fileID = fopen(strcat(mydir,'/','housing-data.txt'),'r');
A = fscanf(fileID,'%f');
fclose(fileID);
AM = vec2mat(A,14);
t = AM(1:355,14);
x = AM(1:355,1:13);
k = 30;
[idx,mu,sum,dist] = kmeans(x,k);
%{
varI = zeros(1,k);
nI = zeros(1,k);
for i= 1:size(x,1)
   varI(idx(i)) = varI(idx(i)) + dist(i,idx(i))*dist(i,idx(i));
   nI(idx(i)) = nI(idx(i)) +1;
end
var = zeros(1,k);
for i= 1:k
   var(i) = varI(i)/nI(i);
end
s = mean(var);
if s==0
    s = 0.00001;
end
%}
s = 100 ;

ph = zeros(k,k);
for r = 1:k
    for c = 1:k
        sm = norm(mu(r,:)' - mu(c,:)');
        power = sm/(2*s^2);
        ph(r,c) = exp(-power);
    end
end
D = zeros(size(x,1),k);
for c = 1:k
    for r = 1:size(x,1)
      sm = norm(x(r,:)' - mu(c,:)');
      power = sm/(2*s^2);
      D(r,c) = exp(-power);
    end
end

L = 0;
%WI = inv((D'*D) + L * eye(k)) * D';            %with quadratic regularization%
WI = inv((D'*D) + L * ph) * D';               %with Tikhonov regularization%
W=WI*t;
y = D*W;
Etrain = error(y,t,L,W);
figure(1);
dim = [.7 .2 .1 .1];
scatter(t,y);
xlabel('t_t_r_a_i_n');
ylabel('y(x,w)');
lower = min(min(y),min(t));
higher = max(max(y),max(t));

axis([lower higher lower higher]);
str  = { strcat('k= ',num2str(k)),strcat('L= ',num2str(L)),strcat('sigma= ',num2str(s))};
annotation('textbox',dim,'String',str,'FitBoxToText','on');
title('Scatter plot with multivariate training data');

tval = AM(356:407,14);
xval = AM(356:407,1:13);
D2 = ones(size(xval,1),k);
for c = 1:k
    for r = 1:size(xval,1)
      sm = norm(xval(r,:)' - mu(c,:)');
      power = sm/(2*s^2);
      D2(r,c) = exp(-power);
    end
end
y1 = D2*W;
Eval = error(y1,tval,L,W);

ttest = AM(408:506,14);
xtest = AM(408:506,1:13);
D3 = ones(size(xtest,1),k);
for c = 1:k
    for r = 1:size(xtest,1)
      sm = norm(xtest(r,:)' - mu(c,:)');
      power = sm/(2*s^2);
      D3(r,c) = exp(-power);
    end
end
y2 = D3*W;

figure(2);
dim = [.7 .3 .1 .1];
scatter(ttest,y2);
xlabel('t_t_e_s_t');
ylabel('y(x,w)');
lower = min(min(y2),min(ttest));
higher = max(max(y2),max(ttest));

axis([lower higher lower higher]);
str  = { strcat('k= ',num2str(k)),strcat('L= ',num2str(L)),strcat('sigma= ',num2str(s))};
annotation('textbox',dim,'String',str,'FitBoxToText','on');
title('Scatter plot with multivariate test data');
Etest = error(y2,ttest,L,W);

function rms = error(y,t,L,W)
    rms = 0;
    for c = 1:length(t)
        rms = rms+(y(c)-t(c))^2;
    end
    rms = rms / length(t);
    rms = rms^(1/2);
    rms = rms + L/2 * dot(W,W);
end