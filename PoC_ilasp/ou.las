%% ----- background knowledge
% we know nothing

%% ----- hypothesis space
#modeh(noop).
#modeh(jump).
#modeh(long_jump).
#modeh(right).
#modeh(jump).
#modeh(sprint).

#modeb(close).
#modeb(far).
#modeb(adjacent).

%% ----- examples will be appended here
#pos({sprint},{},{adjacent. }).
#pos({right},{},{far. }).
#pos({sprint},{},{far. }).
#pos({long_jump},{},{far. }).
#pos({long_jump},{},{adjacent. }).
#pos({long_jump},{},{close. }).
#neg({jump},{},{close. }).
#neg({right},{},{adjacent. }).
#neg({jump},{},{far. }).
