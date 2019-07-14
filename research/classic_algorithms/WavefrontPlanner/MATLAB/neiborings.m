function [row col] = neiborings(r,c, sr, sc)

row = [];
col = [];

% new_r = r;
% new_c = c;


new_r = r + 1;
new_c = c;
if new_r > sr
  new_r = sr;  
end 
row = [row new_r];
col = [col new_c];

new_r = r - 1;
new_c = c;
if new_r < 1
  new_r = 1;    
end
row = [row new_r];
col = [col new_c];

new_r = r;
new_c = c + 1;
if new_c > sc
  new_c = sc;  
end    
row = [row new_r];
col = [col new_c];

new_r = r;
new_c = c - 1;
if new_c < 1
  new_c = 1;
end
row = [row new_r];
col = [col new_c];

new_r = r + 1;
new_c = c + 1;
if new_c > sc
  new_c = sc;  
end
if new_r > sr
  new_r = sr;  
end 
row = [row new_r];
col = [col new_c];

new_r = r - 1;
new_c = c - 1;
if new_c < 1
  new_c = 1;
end
if new_r < 1
  new_r = 1;    
end
row = [row new_r];
col = [col new_c];

new_r = r + 1;
new_c = c - 1;
if new_c < 1
  new_c = 1;
end
if new_r > sr
  new_r = sr;  
end 
row = [row new_r];
col = [col new_c];

new_r = r - 1;
new_c = c + 1;
if new_c > sc
  new_c = sc;  
end
if new_r < 1
  new_r = 1;    
end
row = [row new_r];
col = [col new_c];
