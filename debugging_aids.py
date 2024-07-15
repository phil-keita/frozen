"""This file contains methods to get human-readable strings for 
   the actions, states, and policies of the lake and taxi domains.

   These are to help you debug your code if it is not learning correctly.
"""

from q_values import QValues

def get_lake_action(act: int):
    if act == 0:
        glyph = '<'
    elif act == 1:
        glyph = 'V'
    elif act == 2:
        glyph = '>'
    elif act == 3:
        glyph = '^'
    else:
        raise ValueError(f'Invalid lake action: {act}')
    return glyph

def print_lake_policy(q_vals: QValues):
    NUM_ROWS = 4
    NUM_COLS = 4

    for r in range(NUM_ROWS):
        for c in range(NUM_COLS):
            state = r * NUM_COLS + c

            act = q_vals.best_action(state)
            
            print(get_lake_action(act), end='')
        print('')
    print('')


def print_lake_q_vals(q_vals: QValues):
    NUM_ROWS = 4
    NUM_COLS = 4

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
    
    for r in range(NUM_ROWS):
        print("---------------------------------------------")

        for c in range(NUM_COLS):
            state = r * NUM_COLS + c

            print(f'    {q_vals.q(state, UP):.2f}     |', end='')
        print('')

        for c in range(NUM_COLS):
            state = r * NUM_COLS + c

            print(f' {q_vals.q(state, LEFT):.2f}   {q_vals.q(state, RIGHT):.2f} |', end='')
        print('')
        
        
        for c in range(NUM_COLS):
            state = r * NUM_COLS + c

            print(f'    {q_vals.q(state, DOWN):.2f}     |', end='')
        print('')
            
        print("---------------------------------------------")
    print('')

def get_taxi_action(act: int):
    if act == 0:
        glyph = 'V'
    elif act == 1:
        glyph = '^'
    elif act == 2:
        glyph = '>'
    elif act == 3:
        glyph = '<'
    elif act == 4:
        glyph = 'P'
    elif act == 5:
        glyph = 'D'
    else:
        raise ValueError(f'Invalid taxi action: {act}')
    return glyph

def get_taxi_state_tuple(obs: int):
    res: list[Any] = []

    dest_code = obs % 4
    if dest_code == 0:
        res.append('Dest_R')
    elif dest_code == 1:
        res.append('Dest_G')
    elif dest_code == 2:
        res.append('Dest_Y')
    else:
        res.append('Dest_B')
    obs //= 4

    pass_code = obs % 5
    if pass_code == 0:
        res.append('R')
    elif pass_code == 1:
        res.append('G')
    elif pass_code == 2:
        res.append('Y')
    elif pass_code == 3:
        res.append('B')
    else:
        res.append('T')
    obs //= 5

    coord = obs % 5
    res.append(coord)
    obs //= 5

    res.append(obs)

    res.reverse()
    return res


def print_taxi_policy(q_vals: QValues):
    NUM_ROWS = 5
    NUM_COLS = 5
    NUM_LOCS = 5

    for pass_loc in range(NUM_LOCS):
        loc_str = ''
        if pass_loc == 0:
            loc_str = 'R'
        elif pass_loc == 1:
            loc_str = 'G'
        elif pass_loc == 2:
            loc_str = 'Y'
        elif pass_loc == 3:
            loc_str = 'B'
        else:
            loc_str = 'T'
        print(f'Passenger loc = {loc_str}')

        for r in range(NUM_ROWS):
            for c in range(NUM_COLS):
                state = ((r * 5 + c) * 5 + pass_loc) * 4  # + 0 for destination of R
                
                act = q_vals.best_action(state)
            
                print(get_taxi_action(act), end='')
            print('')
        print('')

    print('\n')

def print_policy(env_name: str, q_vals: QValues):
    if env_name.startswith('lake'):
        print_lake_policy(q_vals)
    elif env_name.startswith('taxi'):
        print_taxi_policy(q_vals)
    else:
        raise ValueError(f"Unrecognized environment name: {env_name}")
    