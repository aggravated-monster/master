%% open veld
%platform(0,255,207,240). mario(41,54,192,208).
%% open veld, mario spring
%platform(0,254,207,240). mario(84,101,181,197).
%% zwevende platforms, goomba op rechts
%enemy(138,154,192,209). platform(75,91,143,159). platform(2,255,208,239). platform(202,219,144,159). platform(139,155,143,159). platform(171,187,144,159). mario(113,126,193,208).
%% pipe en goomba op rechts
%enemy(191,207,192,209). platform(132,163,160,207). platform(0,3,175,208). platform(0,254,207,240). mario(111,126,192,208).
%% pipe, goomba goomba op rechts
%enemy(236,252,192,208). enemy(213,230,192,209). platform(15,46,159,207). platform(143,175,143,208). platform(0,254,207,240). mario(113,130,180,197).

%% Given the relative simplicity of things to avoid in the game,
%% we can generalize enemies and platforms
avoidable_object(XMIN1,XMAX1,YMIN1,YMAX1) :- platform(XMIN1,XMAX1,YMIN1,YMAX1).
avoidable_object(XMIN1,XMAX1,YMIN1,YMAX1) :- enemy(XMIN1,XMAX1,YMIN1,YMAX1).

% Obstacles are always to the right of Mario
%% any avoidable_object Mario cannot walk under is an obstacle (read: anything Mario can bump into
%% given Mario's current Y-plane
%% 1. the avoidable_object box is included in the mario box (small obstacles)
obstacle(XMIN2,XMAX2) :- avoidable_object(XMIN2,XMAX2,YMIN2,YMAX2),
                            mario(XMIN1,XMAX1,YMIN1,YMAX1),
                            XMIN1 < XMIN2,
                            YMIN1 <= YMIN2,
                            YMAX1 >= YMAX2.

%% 2. the avoidable_object box overlaps with the mario box but sticks out above. These are usually high obstacles like pipes
obstacle(XMIN2,XMAX2) :- avoidable_object(XMIN2,XMAX2,YMIN2,YMAX2),
                            mario(XMIN1,XMAX1,YMIN1,YMAX1),
                            XMIN1 < XMIN2,
                            YMIN1 <= YMAX2,
                            YMIN1 > YMIN2.

%% 3. the avoidable_object box overlaps with the mario box but sticks out below.
%% Not sure there are actual examples. Perhaps when Mario is jumping
obstacle(XMIN2,XMAX2) :- avoidable_object(XMIN2,XMAX2,YMIN2,YMAX2),
                            mario(XMIN1,XMAX1,YMIN1,YMAX1),
                            XMIN1 < XMIN2,
                            YMAX1 < YMAX2,
                            YMAX1 >= YMIN2.

%% holes on the other hand require a slightly different approach
%% They are never smaller than Mario
%% And because of detection imprecision, they can be below Mario's Y-plane
%% And lastly, they can be to Mario's left and right, as Mario can fall in
%% 1. Mario is in front of the hole
obstacle(XMIN2,XMAX2) :- hole(XMIN2,XMAX2,YMIN2,YMAX2),
                            mario(XMIN1,XMAX1,YMIN1,YMAX1),
                            XMAX1 < XMIN2,
                            YMAX1 < YMAX2,
                            YMAX1+5 >= YMIN2.

%% 2. Mario has fallen in the hole
obstacle(XMIN2,XMAX2) :- hole(XMIN2,XMAX2,YMIN2,YMAX2),
                            mario(XMIN1,XMAX1,YMIN1,YMAX1),
                            XMIN1 >= XMIN2,
                            XMIN1 < XMAX2,
                            YMAX1 < YMAX2,
                            YMAX1 >= YMIN2.


%% Mario only reacts on what is right in front of him
%% So, in order to determine which obstacle is closest, we can position them relative to each other and the infer the left-most one
left(obstacle(XMIN1,XMAX1),obstacle(XMIN2,XMAX2)) :- obstacle(XMIN1,XMAX1),
                                                        obstacle(XMIN2,XMAX2),
                                                        obstacle(XMIN1,XMAX1) != obstacle(XMIN2,XMAX2),
                                                        XMIN1 < XMIN2.

leftmost(obstacle(XMIN1,XMAX1)) :- obstacle(XMIN1,XMAX1),
                                     not left(_,obstacle(XMIN1,XMAX1)).

close(obstacle(XMIN2,XMAX2)) :- leftmost(obstacle(XMIN2,XMAX2)),
                    mario(XMIN1,XMAX1,YMIN1,YMAX1),
                    |XMIN2-XMAX1| < 20.

close(obstacle(XMIN2,XMAX2)) :- leftmost(obstacle(XMIN2,XMAX2)),
                    mario(XMIN1,XMAX1,YMIN1,YMAX1),
                    hole(XMIN2,XMAX2,YMIN2,YMAX2),
                    XMIN1 >= XMIN2,
                    XMIN1 < XMAX2.

far(obstacle(XMIN2,XMAX2)) :- leftmost(obstacle(XMIN2,XMAX2)),
                    mario(XMIN1,XMAX1,YMIN1,YMAX1),
                    XMIN2-XMAX1 >= 20.