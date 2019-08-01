clc;
clearvars;
mydir = '/home/neha/workspace/data_assign1_group12/';
fileID = fopen(strcat(mydir,'/','train1000.txt'),'r');
A = fscanf(fileID,'%f');
fclose(fileID);
x1 = A(1:3:end);
x2 = A(2:3:end);
t = A(3:3:end);
M = 20;
MN = factorial(M+2)/(factorial(M)*2);
D = zeros(length(x1),M);
[u,v] = meshgrid(0:M);
uv = [u(:),v(:)];
uv(sum(uv,2)>M,:) = [];

for c = 1:MN
    for r = 1:length(x1)
        D(r,c) = x1(r)^(uv(c,1))*x2(r)^(uv(c,2));
    end
end

L = 0;
WI = inv( (D'*D) + L*eye(MN) ) * D';
W=WI*t;
y = D*W;
dim = [.6 .4 .4 .5];
figure(1);
scatter(t,y);
xlabel('t_t_r_a_i_n');
ylabel('y(x,w)');
lower = min(min(y),min(t));
higher = max(max(y),max(t));

axis([lower higher lower higher]);
str  = { strcat('M= ',num2str(M)),strcat('L= ',num2str(L))};
annotation('textbox',dim,'String',str,'FitBoxToText','on');
title('Scatter plot with train data');
%hold on;
%plot(Xtrain,Ytrain,'color','red','LineWidth',2);
%hold on;
Etrain = error(y,t,L,W);

xlin = linspace(min(x1),max(x1),40);
ylin = linspace(min(x2),max(x2),40);
[Xaxis,Yaxis] = meshgrid(xlin,ylin);
f = scatteredInterpolant (x1,x2,y);
Z = f(Xaxis,Yaxis);

figure(2);
mesh(Xaxis,Yaxis,Z)
axis tight; hold on
plot3(x1,x2,t,'.','MarkerSize',15,'MarkerEdgeColor','red');
xlabel('x1');
ylabel('x2');
zlabel('y(x1,x2)');
title(['Plot of (x,y) vs f(x,y,w) for M = ',num2str(M),' L = ',num2str(L)]);
legend('Predicted Model','original output(t_t_r_a_i_n)');

fileID = fopen(strcat(mydir,'/','val.txt'),'r');
A1 = fscanf(fileID,'%f');
fclose(fileID);
x3 = A1(1:3:end);
x4 = A1(2:3:end);
t1 = A1(3:3:end);
D2 = zeros(length(x3),MN);

for c = 1:MN
    for r = 1:length(x3)
        D2(r,c) = x3(r)^(uv(c,1))*x4(r)^(uv(c,2));
    end
end
y1 = D2*W;

Eval = error(y1,t1,L,W);

fileID = fopen(strcat(mydir,'/','test.txt'),'r');
A1 = fscanf(fileID,'%f');
fclose(fileID);
x5 = A1(1:3:end);
x6 = A1(2:3:end);
t2 = A1(3:3:end);
D3 = zeros(length(x5),MN);

for c = 1:MN
    for r = 1:length(x5)
        D3(r,c) = x5(r)^(uv(c,1))*x6(r)^(uv(c,2));
    end
end
y2 = D3*W;
figure(3);
scatter(t2,y2);
xlabel('t_t_e_s_t');
ylabel('y(x,w)');
lower = min(min(y2),min(t2));
higher = max(max(y2),max(t2));

axis([lower higher lower higher]);
annotation('textbox',dim,'String',str,'FitBoxToText','on');
title('Scatter plot with test data');
Etest = error(y2,t2,L,W);

function rms = error(y,t,L,W)
    rms = 0;
    for c = 1:length(t)
        rms = rms+(y(c)-t(c))^2;
    end
    rms = rms / length(t);
    rms = rms^(1/2);
    rms = rms + L/2 * dot(W,W);
end
