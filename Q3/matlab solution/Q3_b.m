function [train_accuracy,val_accuracy]=Q3_b(img_size)
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
%% read val file
val_file = dir('./group_4/val/');
val_file = val_file(3:end);
% val_images = zeros(img_size*img_size,size(val_file,1));
% val_labels = zeros(1,size(val_file,1));
for i=1:size(val_file,1)
    val_filename = val_file(i).name;
    C = strsplit(val_filename,'_');
    val_labels(i) = str2double(cell2mat(C(2)));
    file_path = ['./group_4/val/',val_filename] ;
    image = imread(file_path);
    image = imresize(image,[img_size,img_size]);
    val_images(:,i)=reshape(image,[size(image,1)*size(image,2),1]);
end
images = [train_images, val_images];
labels = [train_labels, val_labels];
%%
net = perceptron;
net.performFcn = 'mse';
net.trainParam.epochs=1;
% net.divideFcn = 'divideind';
% net.divideParam.trainInd=1:size(train_file,1);
% net.divideParam.testInd=size(train_file,1)+1:size(train_file,1)+1+size(val_file,1);

for i=1:500
    net = train(net,double(train_images),train_labels);
    train_accuracy(i) = 1 - mean(abs(train_labels-net(train_images)));
%     net = train(net,train_images,train_labels);
    val_accuracy(i) = 1 - mean(abs(val_labels-net(val_images)));
    if train_accuracy(i) ==1 
        break
    end
end
plot(1:i,train_accuracy);
hold on
plot(1:i,val_accuracy);
disp(max(val_accuracy));
end


