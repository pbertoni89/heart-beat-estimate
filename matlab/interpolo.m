clear all; close all; clc;

xi = 0; xf = 10;
f0 = 4;

N = 10*(xf-xi);
N1 = 2*N;
x = linspace(xi,xf,N);
x1 = linspace(xi,xf,N1);
y = sin(2*pi*x/4);

y1 = spline(x,y,x1);

Y = dft_N(y,N);
Y1 = dft_N(y1,N1);

figure(1);
stem(x,y,'b','filled'); hold on;
stem(x1,y1,'g');

figure(2);
stem(x,Y,'b','filled'); hold on;
stem(x1,Y1,'g'); axis([0 1 -60 80])