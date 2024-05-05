%% ----- background knowledge
% we know nothing

%% ----- hypothesis space
2 ~ noop :- far.
2 ~ right :- far.
2 ~ sprint :- far.
2 ~ jump :- far.
2 ~ long_jump :- far.
2 ~ noop :- close.
2 ~ right :- close.
2 ~ sprint :- close.
2 ~ jump :- close.
2 ~ long_jump :- close.
2 ~ noop :- adjacent.
2 ~ right :- adjacent.
2 ~ sprint :- adjacent.
2 ~ jump :- adjacent.
2 ~ long_jump :- adjacent.
2 ~ noop :- adjacent.

2 ~ noop :- not far.
2 ~ right :- not far.
2 ~ sprint :- not far.
2 ~ jump :- not far.
2 ~ long_jump :- not far.
2 ~ noop :- not close.
2 ~ right :- not close.
2 ~ sprint :- not close.
2 ~ jump :- not close.
2 ~ long_jump :- not close.
2 ~ noop :- not adjacent.
2 ~ right :- not adjacent.
2 ~ sprint :- not adjacent.
2 ~ jump :- not adjacent.
2 ~ long_jump :- not adjacent.

3 ~ noop :- not far, not close.
3 ~ noop :- not far, not adjacent.
3 ~ noop :- not close, not adjacent.
3 ~ right :- not far, not close.
3 ~ right :- not far, not adjacent.
3 ~ right :- not close, not adjacent.
3 ~ sprint :- not far, not close.
3 ~ sprint :- not far, not adjacent.
3 ~ sprint :- not close, not adjacent.
3 ~ jump :- not far, not close.
3 ~ jump :- not far, not adjacent.
3 ~ jump :- not close, not adjacent.
3 ~ long_jump :- not far, not close.
3 ~ long_jump :- not far, not adjacent.
3 ~ long_jump :- not close, not adjacent.

