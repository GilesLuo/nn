function [ net, accu_train, accu_val ] = train_seq( n, images, labels, train_num, val_num, epochs ) 
    images_c = num2cell(images, 1);
    labels_c = num2cell(labels, 1); 

    net = patternnet(n); 
    net.divideFcn = 'dividetrain';
    net.performParam.regularization = 0.25; 
    net.trainFcn = 'traingdx';  % 'trainrp' 'traingdx'     
    net.trainParam.epochs = epochs; 
    accu_train = zeros(epochs,1); 
    accu_val = zeros(epochs,1); 
    for i = 1 : epochs 
           display(['Epoch: ', num2str(i)]);
           idx = randperm(train_num); 
           net = adapt(net, images_c(:,idx), labels_c(:,idx)); 
           pred_train = round(net(images(:,1:train_num)));
           accu_train(i) = 1 - mean(abs(pred_train-labels(1:train_num)));
           pred_val = round(net(images(:,train_num+1:end))); 
           accu_val(i) = 1 - mean(abs(pred_val-labels(train_num+1:end))); 
     end
end

