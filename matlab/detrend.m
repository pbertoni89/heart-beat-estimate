%function [ z_stat ] = detrend(z)
function detrend()
%DETREND1 Summary of this function goes here
%   Detailed explanation goes here
close all; clear; clc;
N = 2000;

lin = (1/N:1/N:1);
r = rand(1,N); g = randn(1,N);
z= r + cos(2*pi*lin) + 2*sin(10*pi*lin) + 4*lin + g;
arm = cos(2*pi*lin) + 2*sin(10*pi*lin);
meanzero = mean(arm);
	
exe = z - 4*lin;

lambda = 10;
I = speye(N);
D2 = spdiags(ones(N-2,1)*[1 -2 1],[0:2],N-2,N);

t1 = tic;
z_s = (I-inv(I+lambda^2*D2'*D2))*z';
z_s = z_s';
toc(t1);

%plot(lin,z,'b'); hold on; pause;
%plot(lin,zok,'r'); pause;            % ->0
plot(lin,z_s,'g'); hold on;           % = z
plot(lin,exe,'c');
plot(lin,arm,'k');
%plot(lin,exenoise-z_s,'k');
%legend('z','zok','z_s','exe','exenoise');
end