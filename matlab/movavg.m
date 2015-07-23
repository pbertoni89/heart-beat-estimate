clear all; close all; clc;

P = 5;
N = 5; % 20

MA = ones(1,P)/P;
MADFT = dft_N(MA,N)';

symm = -floor(P/2):floor(P/2);
dx = 0:N-1;

figure(1)
stem(symm,MA,'b','filled');
axis([ -P/2 P/2 -0.1 0.3]);
title('$$h(n)$$','Interpreter','latex','FontSize',14);
xlabel('$$n$$','Interpreter','latex','FontSize',14);
%title('5-Moving Average Convolution Kernel','FontSize',16);
%______________________________________________________________
figure(2)
stem(dx,abs(MADFT),'g', 'filled');   
axis([ -1 N -0.3 1.1]);
title('$$|H(k)|$$','Interpreter','latex','FontSize',14);
xlabel('$$k\frac{f_c}{10}$$','Interpreter','latex','FontSize',14);
%title('5-Moving Average Amplitude Specter','FontSize',16);
%______________________________________________________________
figure(3)
stem(dx,angle(MADFT),'g', 'filled');   
axis( [ -1 N -2.7 0.9]);
title('$$\angle H(k)$$','Interpreter','latex','FontSize',14);
xlabel('$$k\frac{f_c}{10}$$','Interpreter','latex','FontSize',14);
%title('5-Moving Average Phase Specter','FontSize',16);