## Hopefully easier than the C++ version I was in the middle of.
##
## Masado Ishii  --  UofU SoC, 2018-11-15

## 2018-11-16
## ----------
##   It is relatively straightforward to enumerate the possible paths for
## a single level of refinement. It is possible to count the minimum
## number of orientations contained in each of these paths. My program
## currently does both of these tasks. However, this is not the same as
## finding the minimum number of orientations needed overall,
## since it only captures the complexity of a single level of refinement.
##
##   What we are after is the smallest orientation table that is closed
## under multiplication.
##
##   Note that by unique 'orientation' I mean a pair:
## (region starting node, curve net displacement).
## Such a pair determines the connectivity of the curve segment and 
## the volume filled. Different choices in the hyperplane are left flexible.
##
##   With that clarification, our problem can be formulated as a graph problem. 
## The graph to consider is a bipartite graph. On the left is the set
## of orientations. On the right is a set of groupings of orientations.
## An edge from an orientation (left) to a grouping (right) means
## one of the ways to subdivide that orientation is through the grouping on the
## right. Edges from a grouping (right) to orientations (left) mean
## the orientations are contained in that grouping. In my program thus far,
## an initial orientation on the left is chosen, namely (0, +highest_dim),
## and all groupings which receive an edge from this orientation are explored,
## and we count the number of orientations contained in each grouping. The
## larger problem is to find a minimal orientation table that is closed under
## multiplication. In terms of our graph, we want to select a subset of
## groupings on the right, subject to two constraints. The closure constraint
## is: Every orientation in the range of the subset of groupings must have an
## edge back into the same subset of groupings. 'Range' here means all the
## orientations which receive an edge from any one in the subset of groupings.
## The minimization constraint is: The range of the subset of groupings must
## be minimal.
##





import collections
import itertools

__DEBUG__=False;

def __main__():
    ##dim = 2;   ## The answer it gives is exactly 1 possible solution (correct).
    dim = 3;   ## The answer it gives is 5 possible solutions.
    ##dim = 4;   ## The answer it gives is 5733 possible solutions.
    ##dim = 5;   ## Runs forever.
    (num_solutions, min_num_repl, min_lines, max_num_repl, max_lines) = traverse(dim);
    print("Total # of valid displacement sequences: %d." % num_solutions);
    print("Minimum # of orientations (%d) on lines" % min_num_repl, min_lines);
    print("Maximum # of orientations (%d) on lines" % max_num_repl, max_lines);

def generate_base_path(dim):
    if (dim == 0):
        return [];
    else:
      lower_level = generate_base_path(dim-1);
      return lower_level + [1 << (dim-1)] + lower_level;

def generate_moves(inregion_pos, outregion_pos, region_disp, dim):
    if ((inregion_pos & region_disp) == (outregion_pos & region_disp)):
        return [region_disp];
    else:
        return (1 << d for d in range(dim-1,-1,-1) if (1 << d) != region_disp);

def traverse(dim):
    my_base_path = generate_base_path(dim);
    my_base_order = (
            [0] + list(itertools.accumulate(my_base_path, lambda x,y : x ^ y)));

    print("Displacement sequences:");

    num_regions = 1 << dim;
    assert num_regions == len(my_base_order), "num_regions does not match region visitation list!";

    inregion_start = 0;
    inregion_end = (1 << (dim-1));  ## Net disp is major dim, both inner and outer.
    num_solutions = 0;

    move_hist = [];
    moves = collections.deque();

    ## Count how many orientations we will need.
    ## NOTE: This does not take into account further iterations, which
    ##       fill out the multiplication table of orientations.
    inregion_hist = [];
    min_num_repl = None;
    min_lines = [];
    max_num_repl = None;
    max_lines = [];

    distance = 0;
    inregion_pos = inregion_start;

    while (True):
        if __DEBUG__: print("      State: d=%d, [out,in] == [%d,%d]" % (distance, my_base_order[distance], inregion_pos), end=' ');
        ## Is region terminal?
        if (distance == num_regions-1):
            ## If yes, and we reached the goal, then record a solution.
            final_move = my_base_path[-1];
            if (inregion_pos == inregion_end ^ final_move):
                if __DEBUG__: print("Applying the final move (0,%d)." % final_move);
                move_hist.append(final_move);
                inregion_hist.append(inregion_pos);
                ##print("%5d  " % num_solutions, move_hist);   ## Comment out to save on i/o

                ## Keep track of min and max num of orientations,
                ## which are determined by net displacement and the
                ## starting corner.
                num_repl = len(set(zip(inregion_hist, move_hist)));
                if (min_num_repl is None or num_repl < min_num_repl):
                    min_num_repl = num_repl;
                    min_lines = [num_solutions];
                elif (num_repl == min_num_repl):
                    min_lines.append(num_solutions);
                if (max_num_repl is None or num_repl > max_num_repl):
                    max_num_repl = num_repl;
                    max_lines = [num_solutions];
                elif (num_repl == max_num_repl):
                    max_lines.append(num_solutions);

                num_solutions += 1;
            else:
                if __DEBUG__: print("Stuck at %d" % inregion_pos);

        else:
            ## Otherwise, we generate new moves.
            new_moves = list(generate_moves(
                    inregion_pos, my_base_order[distance],
                    my_base_path[distance], dim));
            for m in new_moves:
                moves.append((inregion_pos, m, distance));

        ## Apply the next move.
        if (not moves):
            break;
        (inregion_pos, next_move, distance) = moves.pop();
        if __DEBUG__: print("Applying the next move (%d,%d)." % (my_base_path[distance], next_move));
        move_hist[distance:] = [next_move];
        inregion_hist[distance:] = [inregion_pos];
        inregion_pos ^= next_move;
        inregion_pos ^= my_base_path[distance];
        distance += 1;

    print("Base path:", my_base_path, sep='\n');

    return (num_solutions, min_num_repl, min_lines, max_num_repl, max_lines);




__main__();
