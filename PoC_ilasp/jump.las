%% objects need to be completely detached to be right or left of each other
%left(O1,O2) :- xmax(O1, X1), xmin(O2, X2), O1 != O2, X1 < X2.
%right(O1,O2) :- xmin(O1, X1), xmax(O2, X2), O1 != O2, X1 > X2.


% jump when close to gap and gap on the right
#pos({
    jump_right(mario)
},{
%% not sure about exclusions
},{
    %% I'm sort of suspecting that ilasp might have troubles inducing intervals, hence predicate close_enough
    %% close_enough can then be implemented in the positioning regular clingo file.
    %% But maybe I underestimate ilasp :)
    %% Another consideration here is that there is only 1 object closest on the right.
    left(mario,gap), close_enough(mario,gap).
}).

#pos({
    jump_right(mario)
},{
%% not sure about exclusions
},{
    %% symmetry
    right(gap,mario), close_enough(mario,gap).
}).

% do not jump when the object is on the left
%% challenge: what happens when we allow move to the left?
#neg({
    jump_right(mario)
},{},{
    left(gap,mario).
}).

% also do not jump when the object is on the right but not close enough
%% by the way: this is going to be a challenge in case of enemies, cuz there can be multiple
#neg({
    jump_right(mario)
},{},{
    right(gap,mario), not close_enough(mario,gap).
}).

% also do not jump when no gap is detected on the right
#neg({
    jump_right(mario)
},{},{
    not right(gap,mario).
}).