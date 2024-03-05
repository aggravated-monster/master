left(O1,O2) :- xmin(O1, X1), xmin(O2, X2), O1 != O2, X1 < X2.
right(O1,O2) :- xmin(O1, X1), xmin(O2, X2), O1 != O2, X1 > X2.

