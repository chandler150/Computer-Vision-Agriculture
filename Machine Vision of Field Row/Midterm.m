%% Midterm 1 for BRAE-428
% By: Chandler Jones
% For Dr. Bo Liu
% 5.4.2020
clc;
clear;
close all;
format short;


%% Question 1

% Two vibration sensors were placed on top of two gearboxes 
%(one healthy condition and one broken-tooth condition) 
%to record vibration data. 
%The data was collected at 30 Hz with no load on the gearboxes.

%Write a MATLAB program to analyze the data of these two gearboxes:

%1.	Plot the vibration data from both gearboxes in the time domain 
    %(x is seconds, y is mm)
%2.	Filter the noise signals from the collected data. 
    %We know that gearbox vibration frequencies are higher than 5 Hz.
%3.	What is the difference between the peak frequencies 
    %of the two vibration signals?

%Comment your program and put it in a Word document. 
%Your program should run without any problems when the Run button is pressed.


%Import Data
brokendata = csvread("MidtermData2021\MidtermData2021\Problem1\BrokenGearbox30HzNoLoad.csv");
healthydata = csvread("MidtermData2021\MidtermData2021\Problem1\HealthyGearbox30HzNoLoad.csv");

%Info about data
size(brokendata);
size(healthydata);

%Generate corresponding time of each sample for each dataset
time = 0;
timearray = zeros(88320, 1);
for i = (1:88320)
    timearray1(i,1) = time;
    
    time = time + 1/30;
end
time = 0;
for i = (1:85873)
    timearray2(i,1) = time;
    
    time = time + 1/30;
end

%Plot default outputs
figure();
subplot(2,2,1);
plot(timearray1,brokendata);
title('Broken Gearbox Sampled at 30Hz, with No Load');
xlabel('Time [s]');
ylabel('Vibration sensor Output [mm]');


subplot(2,2,2);
% 
plot(timearray2,healthydata);
title('Healthy Gearbox Sampled at 30Hz, with No Load');
xlabel('Time [s]');
ylabel('Vibration sensor Output [mm]');
sgtitle('Vibration Data of Two Motors');
%Eliminate the Noise

% Now Filter anything > 5Hz (corresponding to gearbox vibrations)

brokenfiltered = highpass(brokendata,5,30);
healthyfiltered = highpass(healthydata,5,30);

%Plot Filtered Broken Vs Unfiltered Broken


subplot(2,2,3);
hold on;
plot(timearray1,brokendata);
plot(timearray1,brokenfiltered);

legend({'Unfiltered', 'Filtered'},'Location','southeast');
title('Broken Gearbox With Filter');
xlabel('Time [s]');
ylabel('Vibration sensor Output [mm]');

%Plot Filtered Healthy Vs Unfiltered Healthy

subplot(2,2,4);
hold on;
plot(timearray2,healthydata);
plot(timearray2,healthyfiltered);
legend({'Unfiltered', 'Filtered'}, 'Location','southeast');
title('Healthy Gearbox With Filter');
xlabel('Time [s]');
ylabel('Vibration sensor Output [mm]');

% Find Peak Frequencies of Filtered Data


figure()
pspectrum(healthyfiltered,timearray2)
figure()
pspectrum(brokenfiltered,timearray1)

% The peak frequency of the proken motor is roughly 11.25 Hz. It is
% operating at -7 dB. The peak frequency of the healthy motor is 9hz. It is
% operating at 1.072 Db. It makes sense that the healthy motor would have
% an increased power delivery.


%% Problem 2
%You are asked to work on the computer vision on a strawberry harvester 
%robot. The first task of this computer vision system is to identify the 
%strawberry bed right beneath the robot, so the robot can follow it later. 
%A sample image from the robot’s front camera view is shown below.

%1.	Write a MATLAB program to extract the strawberry bed areas from the 
% sample image. You can do any data manipulation as you wish.

%2.	Find the strawberry bed right beneath the robot
%(assuming the camera is mounted near the front center of the robot)

%3.	Find the angle difference between the robot’s heading 
%and the strawberry bed right beneath the robot.

%4.	How can you use the angle difference found in step 3 to control the 
%robot to follow the strawberry bed?

% Import Image
I = imread('MidtermData2021\MidtermData2021\Problem2.jpg');
imshow(I)
BW = createMask(I); % Generated code using HSV color scheme in Color Thresholder
imshow(BW)
[BW1,info] = filterRegions(BW); %Generated code using 'Image Region Analyzer'
s = info.Orientation;

if s > 0
    angle = 90-s;
elseif s == 0
    angle = 90;
else
    angle = -(180 + s - 90);
end
angle
% Since the Orientation is measured from the positive x axis,
% If the Orientation is negative, we ought to find the relative angle from
% the positive Y axis, this being the forward part of the robot will allow
% us to adjust to follow the line, positive necessitating going right, and
% negative necessitating going left.

%Thus in this case, it makes sense for the craft to understand it needs to
%adjust 20 degrees to the left, to adequately change course and finish the
%drive.

%% Problem 3

%Create a simple (shallow) artificial neural network that learns the model
%of the system (regression) shown below using MATLAB Neutral Net Fitting app. 
%x is the input of the system, and y is the output of the system. 
%The data is attached on Canvas. 

%      X(input)      System      Y(output)

%1.	Plot the performance chart 

%2.	Test the trained network with some new inputs the model has never seen before. 

%3.	The model of the system is very similar to which equation?

%Comment your program and put it in the Word document. Answer the question in the Word document.



%First Import Data

data = load("MidtermData2021\MidtermData2021\Problem3.mat");

% Data Preparation

labels  =data.y;                 

Features = data.x; 

%Test new datapoints
newx = [];
newy = [];
for i = (-50:1:50)
    newoutput = myNeuralNetworkFunction(i);
    newy= [newy; newoutput];
    newx= [newx; i];
end

%Plot new data

plot(newx,newy)
title("Validation of Model on New Data")
% Interestingly, the model does not continue the trend towards infinity
