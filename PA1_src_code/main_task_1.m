clc;
clearvars;
mydir = '/home/neha/workspace/data_assign1_group12/';  %directory containing dataset for task1  
imgSets = imageSet(mydir, 'recursive');  %vector of image sets found through a recursive search
sourceFiles=[];
M=[];
for i=1:length(imgSets) 
    srcFiles=dir(strcat(mydir,imgSets(i).Description,'/*.pgm'));
    for j=1:length(srcFiles)
        srcFiles(j).name = strcat(mydir,imgSets(i).Description,'/',srcFiles(j).name);
    end
    l=length(srcFiles);
    sourceFiles=[sourceFiles;srcFiles];
end

for i = 1 : length(sourceFiles)
    I = imread(sourceFiles(i).name); 
    I=I(:); 
    M(:,i)=I; 
end
%calculating Covariance Matrix%
M=M';
MM = bsxfun(@minus,M,mean(M));
covM = MM'*MM/(size(M,1)-1);
%calculating eigenvectors and eigenvalues%
[V D]=eig(covM);
[c, ind]=sort(diag(D),'descend'); 
SD=diag(c);
SV=V(:,ind);

Z=SV(:,1:10);
%get the random permutation for folders from which images for regeneration is to be selected%
f_name = [4 10 14 18 20 21 22 24 25 26 27 29 34 35 38];
f_name_rand = f_name(randperm(length(f_name)));


sig_ev = [1 10 20 40 80 160, 320 640]; %list of significant eigenvectors%
c_title = ['a' 'b' 'c' 'd' 'e' 'f' 'g' 'h'];

%{ 
	By taking the first five folders from the permuation array,we again choose the images randomly from the folders.
	And reconstruct that image for different eigen values 1, 10, 20, 40, 80, 160, 320 and 640 .
 %}
for i = 1:5
    r_img = randi(10,1);
    str=(strcat(mydir,num2str(f_name_rand(i)),'/',num2str(r_img),'.pgm'));
    %disp(str);
    ori_img = imread(str);
    figure(i);
    subplot(3,3,1);
    imshow(ori_img);
    title('Original');
    for j = 1:length(sig_ev)
        Z=SV(:,1:sig_ev(j));
        buff_img=double(ori_img); 
        buff_img=buff_img(:); 
        Mat_I=Z*Z'; 
        Mat=Mat_I*buff_img; 
        new_img=vec2mat(Mat, 64); 
        new_img=new_img'; 
        subplot(3,3,j+1);
        imshow(uint8(new_img),[0,255]); 
        title(c_title(j));
        %title(j);
        
   end
end

