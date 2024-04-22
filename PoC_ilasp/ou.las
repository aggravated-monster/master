
%% ----- background knowledge
% we know nothing
right :- far.
noop.
%% ----- hypothesis space
#modeh(noop).
#modeh(jump).
#modeh(long_jump).
#modeh(right).
#modeh(jump).
#modeh(sprint).

%% ----- this increases resolution time to about 2 minutes
%#modeb(noop).
%#modeb(jump).
%#modeb(long_jump).
%#modeb(right).
%#modeb(jump).
%#modeb(sprint).

#modeb(close).
#modeb(far).
#modeb(adjacent).

%% ----- examples will be appended here
#pos({right},{},{far. }).
#pos({sprint},{},{close. }).
#pos({jump},{},{far. }).
#pos({right},{},{close. }).
#pos({long_jump},{},{close. }).
#pos({jump},{},{close. }).
#pos({jump},{},{adjacent. }).
#pos({long_jump},{},{far. }).
#pos({long_jump},{},{adjacent. }).
#pos({sprint},{},{far. }).
#neg({sprint},{},{adjacent. }).
