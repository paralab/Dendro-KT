## Hopefully easier than the C++ version I was in the middle of.
##
## Masado Ishii  --  UofU SoC, 2018-11-15

import collections
import itertools

__DEBUG__=False;

def __main__():
    num_solutions = traverse();
    print("Total # of valid displacement sequences: %d." % num_solutions);

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

def traverse():
    dim = 3;   ## The answer it gives is 5 possible solutions.
    ##dim = 4;   ## The answer it gives is 5733 possible solutions.
    my_base_path = generate_base_path(dim);
    my_base_order = (
            [0] + list(itertools.accumulate(my_base_path, lambda x,y : x ^ y)));

    num_regions = 1 << dim;
    assert num_regions == len(my_base_order), "num_regions does not match region visitation list!";

    inregion_start = 0;
    inregion_end = (1 << (dim-1));  ## Net disp is major dim, both inner and outer.
    num_solutions = 0;

    move_hist = [];
    moves = collections.deque();

    distance = 0;
    inregion_pos = inregion_start;

    while (True):
        if __DEBUG__: print("State: d=%d, [out,in] == [%d,%d]" % (distance, my_base_order[distance], inregion_pos), end=' ');
        ## Is region terminal?
        if (distance == num_regions-1):
            ## If yes, and we reached the goal, then record a solution.
            final_move = my_base_path[-1];
            if (inregion_pos == inregion_end ^ final_move):
                num_solutions += 1;
                if __DEBUG__: print("Applying the final move (0,%d)." % final_move);
                move_hist.append(final_move);
                print(move_hist);
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
        inregion_pos ^= next_move;
        inregion_pos ^= my_base_path[distance];
        distance += 1;

    return num_solutions;




__main__();
