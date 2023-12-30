#include "locations.lp".

%% objects need to be completely detached to be right or left of each other
left(O1,O2) :- xmax(O1, X1), xmin(O2, X2), O1 != O2, X1 < X2.
right(O1,O2) :- xmin(O1, X1), xmax(O2, X2), O1 != O2, X1 > X2.
%% in case of even 1 pixel overlap, we consider them x-overlapped
%% case 1: O2 completely overlaps 01
x_overlapped(O1,O2) :- xmin(O1,XMIN1), xmin(O2, XMIN2), O1 != O2, xmax(O1,XMAX1), xmax(O2, XMAX2), XMIN1 >= XMIN2, XMAX1 <= XMAX2.
%% case 2: 01 sticks out on the left
%x_overlapped(O1,O2) :- xmin(O1,XMIN1), xmin(O2, XMIN2), O1 != O2, xmax(O1,XMAX1), xmax(O2, XMAX2), XMIN1 < XMIN2, XMAX1 <= XMAX2, XMAX1 >= XMIN2.
%% case 3: x sticks out on the right
%x_overlapped(O1,O2) :- xmin(O1,XMIN1), xmin(O2, XMIN2), O1 != O2, xmax(O1,XMAX1), xmax(O2, XMAX2), XMAX1 > XMAX2, XMIN1 >= XMIN2, XMIN1 <= XMAX2.
