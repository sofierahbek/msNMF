function mat = myvec2mat(vec, m)

n = (length(vec))/m;

mat = (reshape(vec, m, n));