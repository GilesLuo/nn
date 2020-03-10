%% a
% initialization
w = rand(2,1,'double');
w = w .* 0.5;
x = w(1);
y = w(2);
flow = [];
lr = 0.2;
for epoch=1:10000
    grad_x = 400*x^3+2*x-400*y-2 ; 
    grad_y = 200*y-200*x^2 ;
    x = x - lr*grad_x;
    y = y - lr*grad_y;
    flow(epoch,1)=x;
    flow(epoch,2)=y;
    flow(epoch,3)=grad_x;
    flow(epoch,4)=grad_y;
    flow(epoch,5)=(x-1)^2+100*(y-x^2)^2;
    if (x-1)^2+100*(y-x^2)^2 < 0.0001
        value = (x-1)^2+100*(y-x^2)^2;
        break
    end
    %%
x_=0:0.01:1;
y_=0:0.01:1;
f=zeros(size(x_,2),size(y_,2));



end
for i =1:size(x_,2)
    for j=1:size(y_,2)
        f(i,j) =(x_(i)-1)^2+100*(y_(j)-x_(i)^2)^2;
    end
end

% mesh(x_,y_,f);
hold on
plot3(flow(:,1),flow(:,2), flow(:,5),'r');


