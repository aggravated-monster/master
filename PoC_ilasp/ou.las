#pos({happy(alice)},{}).
#pos({happy(claire)},{}).
#neg({happy(bob)},{}).
#neg({happy(dave)},{}).

lego_builder(alice).
lego_builder(bob).
estate_agent(claire).
estate_agent(dave).
enjoys_lego(alice).
enjoys_lego(claire).

#modeh(happy(var(t1))).
#modeb(lego_builder(var(t1))).
#modeb(enjoys_lego(var(t1))).
#modeb(estate_agent(var(t1))).
