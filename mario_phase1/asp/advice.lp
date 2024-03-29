%:- right, close(_).
%jump :- close(_).
long_jump :- close(_).
long_jump :- adjacent().
jump :- adjacent().
sprint :- not close(_), not adjacent(_).