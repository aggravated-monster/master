enemy(XMIN, XMAX, YMIN, YMAX) :- goomba(XMIN,XMAX,YMIN,YMAX).
enemy(XMIN, XMAX, YMIN, YMAX) :- koopa_troopa(XMIN,XMAX,YMIN,YMAX).
enemy(XMIN, XMAX, YMIN, YMAX) :- piranha_plant(XMIN,XMAX,YMIN,YMAX).

platform(XMIN, XMAX, YMIN, YMAX) :- brick(XMIN,XMAX,YMIN,YMAX).
platform(XMIN, XMAX, YMIN, YMAX) :- ground(XMIN,XMAX,YMIN,YMAX).
platform(XMIN, XMAX, YMIN, YMAX) :- pipe(XMIN,XMAX,YMIN,YMAX).
platform(XMIN, XMAX, YMIN, YMAX) :- surprise_box(XMIN,XMAX,YMIN,YMAX).
platform(XMIN, XMAX, YMIN, YMAX) :- surprise_box_empty(XMIN,XMAX,YMIN,YMAX).

above(stone(XMIN1,XMAX1,YMIN1,YMAX1), stone(XMIN2,XMAX2,YMIN2,YMAX2)) :-  stone(XMIN1,XMAX1,YMIN1,YMAX1), stone(XMIN2,XMAX2,YMIN2,YMAX2), stone(XMIN1,XMAX1,YMIN1,YMAX1) != stone(XMIN2,XMAX2,YMIN2,YMAX2), CX = (XMIN1 + XMAX1) / 2, CY = (YMIN1 + YMAX1) / 2, CY2 = (YMIN2 + YMAX2) / 2, CX > XMIN2, CX < XMAX2, CY < CY2, CY > (YMIN2 - (YMAX1 - YMIN1)).

platform(X1,X2,Y1,Y2) :- stone(X1,X2,Y1,Y2), not above(_,stone(X1,X2,Y1,Y2)).

hole(X,255,YMIN,YMAX) :- ground(XMIN,X,YMIN,YMAX), X = #max{XMAX:ground(_,XMAX,_,_)}, X < 25.
hole(0,X,207,240) :- ground(X,XMAX,YMIN,YMAX), X = #min{XMIN:ground(XMIN,_,_,_)}, X > 5.

hole(XMAX1,XMIN2,YMIN1,YMAX1) :- adjacent(ground(XMIN1,XMAX1,YMIN1,YMAX1), ground(XMIN2,XMAX2,YMIN2,YMAX2)).


-adjacent(ground(XMIN1,XMAX1,YMIN1,YMAX1), ground(XMIN2,XMAX2,YMIN2,YMAX2)) :- ground(XMIN1,XMAX1,YMIN1,YMAX1), ground(XMIN3,XMAX3,YMIN3,YMAX3), ground(XMIN2,XMAX2,YMIN2,YMAX2),
                                                                              ground(XMIN1,XMAX1,YMIN1,YMAX1) != ground(XMIN2,XMAX2,YMIN2,YMAX2), ground(XMIN1,XMAX1,YMIN1,YMAX1) != ground(XMIN3,XMAX3,YMIN3,YMAX3),
                                                                              ground(XMIN2,XMAX2,YMIN2,YMAX2) != ground(XMIN3,XMAX3,YMIN3,YMAX3), XMAX3 < XMIN2, XMIN3 > XMAX1.

adjacent(ground(XMIN1,XMAX1,YMIN1,YMAX1), ground(XMIN2,XMAX2,YMIN2,YMAX2)) :- ground(XMIN1,XMAX1,YMIN1,YMAX1), ground(XMIN2,XMAX2,YMIN2,YMAX2), ground(XMIN1,XMAX1,YMIN1,YMAX1) != ground(XMIN2,XMAX2,YMIN2,YMAX2), XMAX1 < XMIN2,
                                                                              not -adjacent(ground(XMIN1,XMAX1,YMIN1,YMAX1), ground(XMIN2,XMAX2,YMIN2,YMAX2)).




%left(O1,O2) :- xmin(O1, X1), xmin(O2, X2), O1 != O2, X1 < X2.
%right(O1,O2) :- xmin(O1, X1), xmin(O2, X2), O1 != O2, X1 > X2.

