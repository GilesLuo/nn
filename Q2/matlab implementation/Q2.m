clear all
%%
x_full = -3:0.01:3;
x = -1:0.05:1;
y = 1.2 .* sin(pi * x)-cos(2.4.*pi.*x);
y_full = 1.2 .* sin(pi * x_full)-cos(2.4.*pi.*x_full);
x_cell = num2cell(x);
y_cell = num2cell(y);

for i = [50]
net = feedforwardnet(i,'trainbfg');
% net.trainParam.epochs=1000;
net.divideFcn = 'dividetrain';
net.layers{1}.transferFcn ='logsig';
net.layers{2}.transferFcn ='purelin';
% net.trainParam.min_grad=10^-20;
net.trainParam.lr=0.001;
% net.trainParam.goal=0.0001;
% net.trainParam.max_fail = 200;
net.performFcn = 'mse';

% net = configure(net,x,y);
%% sequential model
% 
for j=1:300
    net = adapt(net,x_cell, y_cell);
    pred = net(x);
    loss = sum(abs(pred-y).^2);
    disp(loss);
%     if loss<0.001
%         break
%     end
disp(j);
end
% view(net);
%% batch model

% [net, tr] = train(net,x, y);

pred = net(x_full);
%% show result
plot(x_full,y_full,'Linewidth',1.2);
hold on
plot(x_full,pred,'.-');
hold off
% 

end
