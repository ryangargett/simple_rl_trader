

### How to use the new code base

The code base has been refactored into using an inheritance model to mediate the separation of concerns and version 
control problem that we were potentially going to encounter in the coming days. 

An environment and test harness file has been assigned to each of us. The code now functions differently, we each 
have a child 'Trader' class that inherits from the parent 'Simple Trader' class. What this means is: that we can 
individually work on our own 'reset, step, reward, observation and actions functions'; we can make multiple of them 
using the _get_reward_julian_<number> convention; we can use our own test harness; we can plug any model at any time 
into different environments; we can develop useful and generic functions and provide them for all of us to use by 
adding the function to the parent 'Simple Trader' class. For example, once the technical analysis tool is finished 
development, it will be added to the parent class 'Simple Trader', so that anyone who wants to incorporate it into 
their reward function, can do so free of overhead or ignore it - it just becomes another available property/function 
to use in the child implementation.

Once we have written all the 'reset, step, reward, observation and actions' functions that we can think of, we can 
easily apply each agent to each of them, simply by adjusting the line: from environment_<name> import Trader. This 
is an easy and simple but elegant way to handle four of us working on the same files in the same project and avoiding 
conflicts. 

If you have any questions, ask eric or myself.