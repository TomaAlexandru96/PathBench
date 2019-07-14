% University of New Brunswick
% Path planning based on the wavefront algorithm
% Sajad Saeedi
% Winter 2012

clear all;
close all;
clc;

path_closest = 1;
path_random = 0;
dilate_rad = 4;

load grid_map;     
m = uint16(grid_map);

% mark borders as obstacles
[sr sc] = size(m);  % size of map
m(: , sc) = 1;
m(: , 1) = 1;
m(1, :) = 1;
m(sr, :) = 1;

m_org = m;
m_org_noPath = m;

% start cell
S_r = 237;   % row of the start cell
S_c = 77;    % column of the strat point

% set goal cell 
G_r = 20;
G_c = 244;
index = 2;
rc = G_r;
cc = G_c;

figure('name','map with goal and start points');
imshow(m, [0 1]); colorbar();
hold on
plot(S_c, S_r, 'go')   % satrt point
plot(G_c, G_r, 'rsq')  % goal point


%% dialte the obstacles to avoid getting to close to them
for itr = 1:dilate_rad
  for i = 1:sc
    for j = 1:sr
      if m(j,i) == itr  
          r = [];
          c = [];
        [r c] = neiborings(j, i, sr, sc);
        ns = size(r,2);
          for k = 1:ns
            if m(r(k), c(k)) == 0
              m(r(k), c(k)) = itr+1;
            end
          end
      end
    end
  end
end

m_dilated = m;
figure('name','dilated map');
imshow(m, [0 11]); colorbar();
hold on
plot(S_c, S_r, 'go')  % satrt point
plot(G_c, G_r, 'rsq')  % end point

%% make binary dialted make 
  for i = 1:sc
    for j = 1:sr
      if m(j,i) ~= 0  
         m(j,i) = 1; 
      end
    end
  end

figure('name','binary map');
imshow(m, [0 1]); colorbar();
hold on
plot(S_c, S_r, 'go')  % satrt point
plot(G_c, G_r, 'rsq')  % end point


%% generate wavefront
m(G_r, G_c) = index;
WaveFront = false;

while (1)
   index = index + 1;
   [r c] = neiborings(rc, cc, sr, sc);
   ns = size(r,2);
   cc = [];
   rc = [];
   for i = 1:ns
       if m(r(i), c(i)) == 0
           m(r(i), c(i)) = index;
           rc = [rc r(i)];
           cc = [cc c(i)];
       else
           
       end
       if (r(i) == S_r) && (c(i) == S_c)
         index_max = index;

        %   WaveFront = true;
        %   break;
       end
   
   
       if (r(i) == 240) && (c(i) == 9)
           WaveFront = true;
           break;
       end
       
   end

   if WaveFront == true
      break; 
   end       
end

figure('name', 'wavefront'); 
imshow(m , [0 index]); colorbar
m_wave = m;
%% finding the path; marked by 0

m(S_r, S_c) = 0;
rc = S_r;
cc = S_c;

index_for_fugure = index;
index = index_max;

while(1)
    [r c]=neiborings(rc, cc, sr, sc);
    ns = size(r,2);
   
    if path_closest
       distance = [];
       for dd = 1:ns
         distance(dd) = sqrt((r(dd) - S_r)^2+(c(dd) - S_c)^2); 
       end
    end
    %for i = 1:ns
    while(1)
    
       if path_random 
         i = fix(1 + (ns)*rand); % CHOOSE BY RANDOM
       end
    
    
      if path_closest
         [val id] = min(distance);
         i = (id);
      end
      
      if m(r(i), c(i)) == index-1
           m(r(i), c(i)) = 0;
           m_org(r(i), c(i)) = 1;
           rc = r(i);
           cc = c(i);
          break;
      else
          if path_closest
            distance(i) = 100000;
          end
      end
    %end
    end
    
    if (r(i) == G_r) && (c(i) == G_c)
           break;
       end

    index = index - 1;

end

figure('name', 'wavefron and path'); 
imshow(m , [0 index_max ]); colorbar

hold on;
plot(S_c, S_r, 'go')   % satrt point
plot(G_c, G_r, 'rsq')  % stop point

%% Nice plots
%%%%%%%%%%%%%%%%%%
% the original map
m_org_noPath = m_org_noPath + 1;

for i = 1:sc
  for j = 1:sr
    if m_org_noPath(j,i) == 2  
       m_org_noPath(j,i) = 0; 
    end
  end
end

figure('name', 'wavefront path planning'); hold on
subplot(2,2,1)
imshow(m_org_noPath , [0 1]); % colorbar
title('a: map')
hold on;
plot(S_c, S_r, 'go')   % satrt point
plot(G_c, G_r, 'rsq')  % goal point
%%%%%%%%%%%%%%%%%%
% the dilated map
m_dilated = m_dilated + 1;
for i = 1:sc
  for j = 1:sr
    if m_dilated(j,i) == 2  
       m_dilated(j,i) = 0; 
    end
  end
end

for i = 1:sc
  for j = 1:sr
    if m_dilated(j,i) == 1  
       m_dilated(j,i) = 2;  
    end
  end
end
  
for i = 1:sc
  for j = 1:sr
   if m_dilated(j,i) > 2   
     m_dilated(j,i) = 1;
   end 
  end
end 

hold on
subplot(2,2,2)
imshow(m_dilated , [0 2]); %colorbar
title('b: dilated map')
hold on;
plot(S_c, S_r, 'go')   % satrt point
plot(G_c, G_r, 'rsq')  % goal point
%%%%%%%%%%%%%%%%%%%%%%%%% 
% the wave on the map
subplot(2,2,3)
imshow(m_wave , [0 index_for_fugure]); %colorbar
title('c: wavefront')
hold on;
plot(S_c, S_r, 'go')   % satrt point
plot(G_c, G_r, 'rsq')  % goal point
%%%%%%%%%%%%%%%%%%%%%%%%%%
% the original map + path
m_org_ths = m_org + 1;

  for i = 1:sc
    for j = 1:sr
      if m_org_ths(j,i) == 2  
         m_org_ths(j,i) = 0; 
      end
    end
  end
subplot(2,2,4)
imshow(m_org_ths , [0 1]); %colorbar
title('d: map and path')
hold on;
plot(S_c, S_r, 'go')   % satrt point
plot(G_c, G_r, 'rsq')  % goal point

%% Final results
figure('name','map and path');
imshow(m_org_ths , [0 1]); %colorbar

hold on;
plot(S_c, S_r, 'go', 'LineWidth',2,...
                     'MarkerFaceColor','g',...
                     'MarkerSize',5)  % satrt point
plot(G_c, G_r, 'rsq', 'LineWidth',2,...
                     'MarkerFaceColor','r',...
                     'MarkerSize',5)  % goal point
