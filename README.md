# fiber_coupling_mat292

This is project code for a fiber coupling optimization system. A demonstration video of the setup can be viewed at LINK.

To satify code runnability for the assignment, we simulated the power function as a rosenbruck function, as that is similar to the predicted power function. The TA does not have access to the hardware (they are welcome to come into the lab though), so we provided the simulated power function with approximtely gaussian noise. 

To run the files, first the adaptive gradient descent function should be called to a satisfactory tolerance (SOME MULTIPLE OF NOISE). Once a location is reached, nedler mead should be run (ideally more than once in case it gets stuck at a local maxima).

The optimal point was first reached using adaptive gradient descent. Once a rough location of the maxima was determined, we used nedler mead to determine a closer point. 
