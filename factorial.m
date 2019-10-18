function fact = factorial(a)
fact = ones(1,1);
while a > 1,
fact = fact*(a);
a = a-1;
end;