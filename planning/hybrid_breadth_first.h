#ifndef HYBRID_BREADTH_FIRST_H_
#define HYBRID_BREADTH_FIRST_H_

#include <vector>

using std::vector;

class HBF {
 public:
  // Constructor
  HBF();

  // Destructor
  virtual ~HBF();

  // HBF structs
  struct maze_s {
    int pos_depth; 
    double x;
    double y;
    double theta;
    double heuristic;
  };

  struct maze_path {
    vector<vector<vector<int>>> explored;
    vector<vector<vector<maze_s>>> came_from;
    maze_s final;
  };

  struct statePair{
    maze_s state;
    int position;
  };
  
  // HBF functions
  int theta_to_stack_number(double theta);

  int idx(double float_num);

  vector<maze_s> generatePotentialMovesAndStates(maze_s &state, int &xgoal, int &ygoal);

  vector<maze_s> reconstruct_path(vector<vector<vector<maze_s>>> &came_from, vector<double> &start, HBF::maze_s &final);

  maze_path search(vector<vector<int>> &grid, vector<double> &start, vector<int> &goal);

  double calculateHeuristic(double &xpos, double &ypos, int &xgoal, int &ygoal);

  statePair findLowestCostState(vector<maze_s> &not_explored);

  bool isStateinVector(vector<maze_s> &not_explored, HBF::maze_s &state);

 private:
  const int NUM_THETA_CELLS = 90;
  const double SPEED = 1.45;
  const double LENGTH = 0.5;
};

#endif  // HYBRID_BREADTH_FIRST_H_