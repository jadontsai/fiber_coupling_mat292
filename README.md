## Requirements
- Python 3.6+
- NumPy
- SciPy
- matplotlib

Install dependencies with:
pip install -r requirements.txt

# fiber_coupling_project

This is project code for a fiber coupling optimization system. A demonstration video of the setup can be viewed at [this link](https://youtube.com/shorts/9mXn-IsixLE?feature=share).

To satify code runnability for the assignment, we simulated the power function in power_function.py, as that is similar to the predicted power function as viewd in the lab. The TA does not have access to the hardware (they are welcome to come into the lab though), so we added gaussian noise to the power function to simulate lab conditions. 

To run the files, first the adaptive gradient descent function should be called to a satisfactory tolerance (decision of exact tolerance up to you; if you wanted to use our values you can just call the functions as is as they are the ones currently coded in as parameters and the same as the report ones). Once a rough location is reached, nedler mead should be run to find a more precise location of maxima.
