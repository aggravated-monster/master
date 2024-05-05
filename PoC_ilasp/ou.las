
%% ----- background knowledge
% we know nothin
%% ----- hypothesis space
2 ~  :- right, far.
2 ~  :- right, close.
2 ~  :- right, adjacent.
2 ~  :- sprint, far.
2 ~  :- sprint, close.
2 ~  :- sprint, adjacent.
2 ~  :- jump, far.
2 ~  :- jump, close.
2 ~  :- jump, adjacent.
2 ~  :- long_jump, far.
2 ~  :- long_jump, close.
2 ~  :- long_jump, adjacent.
2 ~  :- noop, far.
2 ~  :- noop, close.
2 ~  :- noop, adjacent.

2 ~  :- not right, far.
2 ~  :- not right, close.
2 ~  :- not right, adjacent.
2 ~  :- not sprint, far.
2 ~  :- not sprint, close.
2 ~  :- not sprint, adjacent.
2 ~  :- not jump, far.
2 ~  :- not jump, close.
2 ~  :- not jump, adjacent.
2 ~  :- not long_jump, far.
2 ~  :- not long_jump, close.
2 ~  :- not long_jump, adjacent.
2 ~  :- not noop, far.
2 ~  :- not noop, close.
2 ~  :- not noop, adjacent.

2 ~  :- right, not far.
2 ~  :- right, not close.
2 ~  :- right, not adjacent.
2 ~  :- sprint, not far.
2 ~  :- sprint, not close.
2 ~  :- sprint, not adjacent.
2 ~  :- jump, not far.
2 ~  :- jump, not close.
2 ~  :- jump, not adjacent.
2 ~  :- long_jump, not far.
2 ~  :- long_jump, not close.
2 ~  :- long_jump, not adjacent.
2 ~  :- noop, not far.
2 ~  :- noop, not close.
2 ~  :- noop, not adjacent.

2 ~  :- not right, not far.
2 ~  :- not right, not close.
2 ~  :- not right, not adjacent.
2 ~  :- not sprint, not far.
2 ~  :- not sprint, not close.
2 ~  :- not sprint, not adjacent.
2 ~  :- not jump, not far.
2 ~  :- not jump, not close.
2 ~  :- not jump, not adjacent.
2 ~  :- not long_jump, not far.
2 ~  :- not long_jump, not close.
2 ~  :- not long_jump, not adjacent.
2 ~  :- not noop, not far.
2 ~  :- not noop, not close.
2 ~  :- not noop, not adjacent.

2 ~  :- right, far.
2 ~  :- right, far.
2 ~  :- edge(V0, V0), in(V1,V2).
2 ~  :- edge(V0, V0), reach(V1).
2 ~  :- edge(V0, V1), in(1,V2).
2 ~  :- edge(V0, V1), reach(V2).

%% ----- examples will be appended here
#pos({},{},{jump. far. }).
#pos({},{},{sprint. far. }).
#pos({},{},{jump. close. }).
#pos({},{},{jump. adjacent.}).
#neg({},{},{sprint. adjacent.}).
#neg({},{},{right. adjacent.}).
