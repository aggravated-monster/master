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

%% Platforms, enemies and holes are all objects
object(XMIN1,XMAX1,YMIN1,YMAX1) :- platform(XMIN1,XMAX1,YMIN1,YMAX1).
object(XMIN1,XMAX1,YMIN1,YMAX1) :- enemy(XMIN1,XMAX1,YMIN1,YMAX1).
object(XMIN1,XMAX1,YMIN1,YMAX1) :- hole(XMIN1,XMAX1,YMIN1,YMAX1).

%% We are only interested in objects located on Mario's right. An object is to the right when
%% its XMIN > Mario's XMAX.
%% Platforms hovering over Mario are therefore not to Mario's right, as he is already walking under them.
right(object(XMIN2,XMAX2,YMIN2,YMAX2)) :- object(XMIN2,XMAX2,YMIN2,YMAX2),
                                          mario(XMIN1,XMAX1,YMIN1,YMAX1),
                                          XMIN2 > XMAX1.

%% Some objects float above Mario, such that he might walk under them
%% An object floats above Mario if its YMAX <  Mario's YMIN
%% Again, we are only interested in such objects if they are on the right
above(object(XMIN2,XMAX2,YMIN2,YMAX2)) :- right(object(XMIN2,XMAX2,YMIN2,YMAX2)),
                                          mario(XMIN1,XMAX1,YMIN1,YMAX1),
                                          YMAX2 < YMIN1.

%% Obstacles are always to the right of Mario and in his Y-plane, so any object to Mario's right and
%% not above him
%% (In this solution, we do not take trajectories into account, meaning we do not look at objects
%% that may become an obstacle when a wrong action is chosen)
obstacle(XMIN2,XMAX2) :- right(object(XMIN2,XMAX2,YMIN2,YMAX2)),
                         not above(object(XMIN2,XMAX2,YMIN2,YMAX2)).


%% Mario only reacts on what is right in front of him
%% So, in order to determine which obstacle is closest, we can position them relative to each other and the infer the left-most one
left(obstacle(XMIN1,XMAX1),obstacle(XMIN2,XMAX2)) :- obstacle(XMIN1,XMAX1),
                                                     obstacle(XMIN2,XMAX2),
                                                     obstacle(XMIN1,XMAX1) != obstacle(XMIN2,XMAX2),
                                                     XMIN1 < XMIN2.

leftmost(obstacle(XMIN1,XMAX1)) :- obstacle(XMIN1,XMAX1),
                                   not left(_,obstacle(XMIN1,XMAX1)).

%% close is when Mario is between 15 and 5 xpos away from an obstacle.
%% anything closer and mario dies anyway, or gets stuck against a pipe because of bizarre
%% emulation behaviour. The only escape then is randomisation
%% so leave that to the RL learner
close(obstacle(XMIN2,XMAX2)) :- leftmost(obstacle(XMIN2,XMAX2)),
                    mario(XMIN1,XMAX1,YMIN1,YMAX1),
                    XMIN2-XMAX1 < 15,
                    XMIN2-XMAX1 >= 5.

adjacent(platform(XMIN2,XMAX2,YMIN2,YMAX2)) :- leftmost(obstacle(XMIN2,XMAX2)),
                                                platform(XMIN2,XMAX2,YMIN2,YMAX2),
                                                mario(XMIN1,XMAX1,YMIN1,YMAX1),
                                                XMIN2-XMAX1 < 5.

far(obstacle(XMIN2,XMAX2)) :- leftmost(obstacle(XMIN2,XMAX2)),
                    mario(XMIN1,XMAX1,YMIN1,YMAX1),
                    XMIN2-XMAX1 >= 20.