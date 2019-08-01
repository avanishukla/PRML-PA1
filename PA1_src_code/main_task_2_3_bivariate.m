clc;
clearvars;
mydir = '/home/neha/workspace/data_assign1_group12/';
fileID = fopen(strcat(mydir,'/','train20.txt'),'r');
A = fscanf(fileID,'%f');
fclose(fileID);
AM = vec2mat(A,3);
t = AM(:,3);
x = AM(:,1:2);
k = 18;     %20(100),12(20),18(20) ovefitting
[idx,mu,sum,dist] = kmeans(x,k);

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

L = 0.00000003;  %0.00000003(20)
WI = inv((D'*D) + L * eye(k)) * D';            %with quadratic regularization%
%WI = inv((D'*D) + L * ph) * D';               %with Tikhonov regularization%
W=WI*t;
y = D*W;
dim = [.7 .4 .4 .5];
figure(1);
scatter(t,y);
xlabel('t_t_r_a_i_n');
ylabel('y(x,w)');
axis([-100 50 -100 50]);
str  = { strcat('S= ',num2str(s)),strcat('L= ',num2str(L))};
annotation('textbox',dim,'String',str,'FitBoxToText','on');
title('Scatter plot with train data');

Etrain = error(y,t,L,W);

xlin = linspace(min(x(:,1)),max(x(:,1)),40);
ylin = linspace(min(x(:,2)),max(x(:,2)),40);
[Xaxis,Yaxis] = meshgrid(xlin,ylin);
f = scatteredInterpolant (x(:,1),x(:,2),y);
Z = f(Xaxis,Yaxis);

figure(2);
mesh(Xaxis,Yaxis,Z) %interpolated
axis tight; hold on
plot3(x(:,1),x(:,2),t,'.','MarkerSize',15,'MarkerEdgeColor','red'); %nonuniform
xlabel('x1');
ylabel('x2');
zlabel('y(x1,x2)');
title(['Plot of (x,y) vs f(x,y,w) for k = ',num2str(k),' L = ',num2str(L)]);
legend('Predicted Model','Actual output(t_t_r_a_i_n)');

fileID = fopen(strcat(mydir,'/','val.txt'),'r');
A = fscanf(fileID,'%f');
fclose(fileID);
AM = vec2mat(A,3);
tval = AM(:,3);
xval = AM(:,1:2);

D2 = zeros(size(xval,1),k);
for c = 1:k
    for r = 1:size(xval,1)
      sm = norm(xval(r,:)' - mu(c,:)');
      power = sm/(2*s^2);
      D2(r,c) = exp(-power);
    end
end
y1 = D2*W;

Eval = error(y1,tval,L,W);

fileID = fopen(strcat(mydir,'/','test.txt'),'r');
A = fscanf(fileID,'%f');
fclose(fileID);
AM = vec2mat(A,3);
ttest = AM(:,3);
xtest = AM(:,1:2);

D3 = zeros(size(xtest,1),k);
for c = 1:k
    for r = 1:size(xtest,1)
      sm = norm(xtest(r,:)' - mu(c,:)');
      power = sm/(2*s^2);
      D3(r,c) = exp(-power);
    end
end
y2 = D3*W;
dim = [.7 .4 .4 .5];
figure(3);
scatter(ttest,y2);
xlabel('t_t_e_s_t');
ylabel('y(x,w)');
axis([-100 100 -100 100]);
str  = { strcat('S= ',num2str(s)),strcat('L= ',num2str(L))};
annotation('textbox',dim,'String',str,'FitBoxToText','on');
title('Scatter plot with test data');
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
