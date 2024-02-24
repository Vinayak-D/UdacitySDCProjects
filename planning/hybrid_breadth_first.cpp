#include <math.h>
#include <iostream>
#include <vector>
#include "hybrid_breadth_first.h"

// Initializes HBF
HBF::HBF() {}

HBF::~HBF() {}

int HBF::theta_to_stack_number(double theta){
  // Takes an angle (in radians) and returns which "stack" in the 3D 
  //   configuration space this angle corresponds to. Angles near 0 go in the 
  //   lower stacks while angles near 2 * pi go in the higher stacks.
  double new_theta = fmod((theta + 2 * M_PI),(2 * M_PI));
  int stack_number = (int)(round(new_theta * NUM_THETA_CELLS / (2*M_PI))) 
                   % NUM_THETA_CELLS;

  return stack_number;
}

int HBF::idx(double float_num) {
  // Returns the index into the grid for continuous position. So if x is 3.621, 
  //   then this would return 3 to indicate that 3.621 corresponds to array 
  //   index 3.
  return int(floor(float_num));
}

//generate all potential moves and their corresponding states from a start state (position / orientation)
vector<HBF::maze_s> HBF::generatePotentialMovesAndStates(HBF::maze_s &state, int &xgoal, int &ygoal) {
  int depth = state.pos_depth;
  double x = state.x;
  double y = state.y;
  double theta = state.theta;
    
  int depth2 = depth + 1;
  vector<HBF::maze_s> next_states;

  //steering angle assumed from -35 to 40 deg in 5 deg increments
  for(double delta_i = -35; delta_i < 40; delta_i+=5) {
    double delta = M_PI / 180.0 * delta_i;
    double omega = SPEED / LENGTH * tan(delta);
    double theta2 = theta + omega;
    if(theta2 < 0) {
      theta2 += 2*M_PI;
    }
    double x2 = x + SPEED * cos(theta);
    double y2 = y + SPEED * sin(theta);
    HBF::maze_s state2;
    state2.pos_depth = depth2;
    state2.x = x2;
    state2.y = y2;
    state2.theta = theta2;
    state2.heuristic = depth2 + calculateHeuristic(x2, y2, xgoal, ygoal);
    next_states.push_back(state2);
  }

  return next_states;
}

vector< HBF::maze_s> HBF::reconstruct_path(
  vector<vector<vector<HBF::maze_s>>> &came_from, vector<double> &start, 
  HBF::maze_s &final) {

  vector<maze_s> path = {final};
  
  int stack = theta_to_stack_number(final.theta);

  maze_s current = came_from[stack][idx(final.x)][idx(final.y)];
  
  stack = theta_to_stack_number(current.theta);
  
  double x = current.x;
  double y = current.y;

  while(x != start[0] || y != start[1]) {
    path.push_back(current);
    current = came_from[stack][idx(x)][idx(y)];
    x = current.x;
    y = current.y;
    stack = theta_to_stack_number(current.theta);
  }
  
  return path;
}

double HBF::calculateHeuristic(double &xpos, double &ypos, int &xgoal, int &ygoal){
  double heuristic = sqrt(pow(xpos-(double)xgoal,2) + pow(ypos-(double)ygoal,2));
  return heuristic;
}

HBF::statePair HBF::findLowestCostState(vector<HBF::maze_s> &not_explored){
  double maxHeuristic = 10000.0;
  maze_s lowestCostState;
  int position;
  for (int i = 0; i < not_explored.size(); i++){
    if (not_explored[i].heuristic < maxHeuristic){
      maxHeuristic = not_explored[i].heuristic;
      lowestCostState = not_explored[i];
      position = i;
    }
  }
  statePair pairReturned;
  pairReturned.state = lowestCostState;
  pairReturned.position = position;
  return pairReturned;
}

bool HBF::isStateinVector(vector<maze_s> &not_explored, HBF::maze_s &state){
  for (int i = 0; i < not_explored.size(); i++){
    maze_s stateCompare = not_explored[i];
    if ((stateCompare.x == state.x) && (stateCompare.y == state.y) && (stateCompare.theta == state.theta)){
      return true;
    }
  }
  return false;
}


HBF::maze_path HBF::search(vector< vector<int> > &grid, vector<double> &start, 
                           vector<int> &goal) {
  vector<vector<vector<int>>> explored(NUM_THETA_CELLS, vector<vector<int>>(grid[0].size(), vector<int>(grid.size())));
  vector<vector<vector<maze_s>>> came_from(NUM_THETA_CELLS, vector<vector<maze_s>>(grid[0].size(), vector<maze_s>(grid.size())));
  double theta = start[2];
  int stack = theta_to_stack_number(theta);
  int depth = 0;

  //initialize start state
  maze_s state;
  state.pos_depth = depth;
  state.x = start[0];
  state.y = start[1];
  state.theta = theta;
  state.heuristic = 0.0;

  //mark start state as explored and add to path
  explored[stack][idx(state.x)][idx(state.y)] = 1;
  came_from[stack][idx(state.x)][idx(state.y)] = state;
  int total_explored = 1;

  //initialize not_explored, a vector / dictionary / list of states, add start state
  vector<maze_s> not_explored = {state};

  bool finished = false;

  //if not explored is empty and goal is not reached, it means obstacles everywhere
  while(!not_explored.empty()) {
    //maze_s current = not_explored[0]; //grab first element
    statePair pairReturned = findLowestCostState(not_explored);
    maze_s current = pairReturned.state;
    not_explored.erase(not_explored.begin()+pairReturned.position); //pop lowest cost state

    int x = current.x;
    int y = current.y;

    //if goal reached, return path
    if(idx(x) == goal[0] && idx(y) == goal[1]) {
      std::cout << "found path to goal in " << total_explored << " expansions" 
                << std::endl;
      maze_path path;
      path.came_from = came_from;
      path.explored = explored;
      path.final = current;

      return path;
    }

    //otherwise get all potential moves and states
    vector<maze_s> next_state = generatePotentialMovesAndStates(current, goal[0], goal[1]);

    //iterate through all potential moves
    for(int i = 0; i < next_state.size(); ++i) {

      //for each potential move
      int depth2 = next_state[i].pos_depth;
      double x2 = next_state[i].x;
      double y2 = next_state[i].y;
      double theta2 = next_state[i].theta;

      //check if valid move or not
      if((x2 < 0 || x2 >= grid.size()) || (y2 < 0 || y2 >= grid[0].size())) {
        // invalid cell
        continue;
      }
      if (grid[idx(x2)][idx(y2)] == 1){
        //wall or obstacle
        continue;
      }

      int stack2 = theta_to_stack_number(theta2);

      //if the state is not present in explored
      if(explored[stack2][idx(x2)][idx(y2)] == 0 && !isStateinVector(not_explored, next_state[i])) {
        //add to the not explored vector
        not_explored.push_back(next_state[i]);
        //mark state as explored
        explored[stack2][idx(x2)][idx(y2)] = 1;
        //mark state in came_from to calculate path later
        came_from[stack2][idx(x2)][idx(y2)] = current;
        ++total_explored;
      }
    }
  }

  std::cout << "no valid path." << std::endl;
  HBF::maze_path path;
  path.came_from = came_from;
  path.explored = explored;
  path.final = state;

  return path;
}