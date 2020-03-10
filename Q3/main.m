clear all
%% read train file
train_file = dir('./group_4/train');
train_file = train_file(3:end);
train_images = zeros(256*256,size(train_file,1));
train_labels = zeros(1,size(train_file,1));
for i=1:size(train_file,1)
    filename = train_file(i).name;
    C = strsplit(filename,'_');
    train_labels(i) = str2double(cell2mat(C(2)));
    file_path = ['group_4/train/',filename] ;
    image = imread(file_path);
    train_images(:,i)=reshape(image,[256*256,1]);
end
%% read val file
val_file = dir('./group_4/val');
val_file = train_file(3:end);
val_images = zeros(256*256,size(val_file,1));
val_labels = zeros(1,size(val_file,1));
for i=1:size(val_file,1)
    filename = val_file(i).name;
    C = strsplit(filename,'_');
    val_labels(i) = str2double(cell2mat(C(2)));
    file_path = ['group_4/train/',filename] ;
    image = imread(file_path);
    val_images(:,i)=reshape(image,[256*256,1]);
end
images = [train_images, val_images];
labels = [train_labels, val_labels];
net = perceptron;
net.performFcn = 'mse';
net.trainParam.epochs=1;
% net.divideFcn = 'divideind';
% net.divideParam.trainInd=1:size(train_file,1);
% net.divideParam.testInd=size(train_file,1)+1:size(train_file,1)+1+size(val_file,1);

for i=1:20
    net = train(net,train_images,train_labels);
    train_accuracy(i) = 1 - sum(abs(train_labels-net(train_images)))/size(train_file,1);
%     net = train(net,train_images,train_labels);
    val_accuracy(i) = 1 - sum(abs(val_labels-net(val_images)))/size(val_file,1);
end
plot(1:20,train_accuracy);
hold on
plot(1:20,val_accuracy);