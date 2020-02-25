%% a
% initialization
w = rand(2,1,'double');
w = w .* 0.5;
x = w(1);
y = w(2);
flow = [];
lr = 0.001;
for epoch=1:10000
    grad_x = 400*x^3+2*x-400*y-2 ; 
    grad_y = 200*y-200*x^2 ;
    x = x - lr*grad_x;
    y = y - lr*grad_y;
    flow(epoch,1)=x;
    flow(epoch,2)=y;
    flow(epoch,3)=grad_x;
    flow(epoch,4)=grad_y;
    if (x-1)^2+100*(y-x^2)^2 < 0.001
        value = (x-1)^2+100*(y-x^2)^2;
        break
    end
end
plot(flow(:,1),flow(:,2));


