#modeh(heads(var(coin))).
#modeh(tails(var(coin))).

#modeb(heads(var(coin))).
#modeb(tails(var(coin))).
#modeb(coin(var(coin))).

#modeh(heads(const(coin))).
#modeh(tails(const(coin))).

#constant(coin, c1).
#constant(coin, c2).
#constant(coin, c3).

coin(c1).
coin(c2).
coin(c3).

#pos({heads(c1), tails(c2), heads(c3)},
     {tails(c1), heads(c2), tails(c3)}).

#pos({heads(c1), heads(c2), tails(c3)},
     {tails(c1), tails(c2), heads(c3)}).