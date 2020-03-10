function [tr_loss,val_loss]=Q3_c(img_size,num_neuron)
%% read train file
train_file = dir('./group_4/train/');
train_file = train_file(3:end);
% train_images = zeros(img_size*img_size,size(train_file,1));
% train_labels = zeros(1,size(train_file,1));
for i=1:size(train_file,1)
    tr_filename = train_file(i).name;
    C = strsplit(tr_filename,'_');
    train_labels(i) = str2double(cell2mat(C(2)));
    file_path = ['./group_4/train/',tr_filename] ;
    image = imread(file_path);
    image = imresize(image,[img_size,img_size]);
    train_images(:,i)=reshape(image,[size(image,1)*size(image,2),1]);
end
train_images = double(train_images);
%% read val file
val_file = dir('./group_4/val/');
val_file = val_file(3:end);
for i=1:size(val_file,1)
    val_filename = val_file(i).name;
    C = strsplit(val_filename,'_');
    val_labels(i) = str2double(cell2mat(C(2)));
    file_path = ['./group_4/val/',val_filename] ;
    image = imread(file_path);
    image = imresize(image,[img_size,img_size]);
    val_images(:,i)=reshape(image,[size(image,1)*size(image,2),1]);
end
val_images = double(val_images);
%%
net = patternnet(num_neuron);
% net.performFcn = 'mse';
net.trainParam.epochs=10;
net.divideFcn = 'divideind';
net.trainParam.min_grad = 10e-20;
net.divideParam.trainInd=1:size(train_file,1);
net.divideParam.testInd=size(train_file,1)+1:size(train_file,1)+1+size(val_file,1);

for i=1:600
    net = train(net,train_images,train_labels);
    y_tr = net(train_images);
    y_val = net(val_images);
    tr_loss(i) = mean(train_labels-y_tr).^2;
    val_loss(i) = mean(val_labels-y_val).^2;
    train_accuracy(i) = 1 - mean(abs(train_labels-y_tr));
    val_accuracy(i) = 1 - mean(abs(val_labels-y_val));
    if train_accuracy(i) ==1 && PCA==false
        break
    end
end
plot(0:10:(i-1)*10,train_accuracy,'Linewidth',1);
hold on
plot(0:10:(i-1)*10,val_accuracy,'Linewidth',1);
disp(max(val_accuracy));
end


