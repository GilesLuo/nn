
w = rand(2,1,'double');
w = w .* 0.5;
flow = [];
lr = 1;
%%
syms x y;
H=[1200*x^2+2 -400;
   -400*x  200];
flow=zeros(10000001,2);
    
for epoch=1:1000
    grad_w = [400*w(1)^3+2*w(1)-400*w(2)-2 ; 
              200*w(2)-200*w(1)^2 ;];
    
    h=subs(H,[x,y],[w(1),w(2)]);
    w = w - inv(double(h)) * grad_w;
    flow(epoch,1)=w(1);
    flow(epoch,2)=w(2);
    flow(epoch,3)=(w(1)-1)^2+100*(w(2)-w(1)^2)^2;
    if abs(w(1)-1)+abs(w(2)-1)<0.0001
        value = (w(1)-1)^2+100*(w(2)-w(1)^2)^2;
        break
    
    end
    disp(epoch);
end
plot3(flow(1:epoch,1),flow(1:epoch,2), flow(1:epoch,3));