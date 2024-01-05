%% ----- background knowledge
% we know nothing

%% ----- hypothesis space
%#modeh(jump(const(hero))).
#modeh(jump).

#modeb(right(var(obj))).
#modeb(close(var(obj))).
#modeb(far(var(obj))).

%% bind constant to atom mario
%#constant(hero, mario).


%% ----- examples

% jump when close to gap and gap on the right
% sonofabitch how the F does this context thing work really
#pos({jump},{far(gap),left(gap)},{right(gap).close(gap).}).
#neg({jump},{},{left(gap).}).
#neg({jump},{},{far(gap).}).


