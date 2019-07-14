# Wavefront Planning

Implementation is based on [1]. This is the description of the algorithm from Chapter 4.5 of [1] - Wave-Front Planner:
"This planner determines a path via gradient descent on the grid starting from the start. Essentially, the planner determines the path one pixel at a time. The wave-front planner essentially forms a potential function on the grid which has one local minimum and thus is resolution complete. The planner also determines the shortest path, but at the cost of coming dangerously close to obstacles. The major drawback of this method is that the planner has to search the entire space for a path"

## How to Run
### Python
Open a terminal type these commands

* cd WavefrontPlanner
* python Python/Wavefront.py 

and follow the instructions.

### MATLAB
Open wavefront.m and hit F5 or press 'Run'.

### Prerequisites for Python
* Python 2.7
* pygame

## Authors

* **Sajad Saeedi** 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## References
[1] H. Choset, K. M. Lynch, S. Hutchinson, G. Kantor, W. Burgard, L. E. Kavraki and S. Thrun, Principles of Robot Motion: Theory, Algorithms, and Implementations,
MIT Press, Boston, 2005, http://www.cs.cmu.edu/afs/cs/Web/People/motionplanning/ 

