function a = af(theta, N)

a = zeros(N, 1);
for l = 1 : N
    a(l) = exp(1j*2*pi*0.5*(l-1)*sin(theta));
end
end