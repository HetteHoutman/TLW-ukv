import iris
import datetime
from iris.time import PartialDateTime

import numpy as np

def read_variable(pp_file, code, hour_selected):
    '''
    Reads variable defined by stash code from pp_file.
    
    Args:
        pp_file (str)
        code (int)
        
    Returns:
        cubes (list)
    '''
    stash_code=iris_stash_code(code)
    stash_const=iris.AttributeConstraint(STASH = stash_code)
    cubes = iris.load(pp_file, stash_const) 
    print(f"Reading data from stash {code:d} at hour {hour_selected:d}")
    hour_const = iris.Constraint(time=lambda cell : 
                                 cell.point.hour == hour_selected)
    cube = cubes.extract(hour_const)[0]
    
    return cube
    
def iris_stash_code(code):
    '''
    Converts stash code to iris format
    
    Args:
        code : Stash code string of up to 5 digits
        
    Returns:
        stash code in iris format
    '''
    temp = f"{code:05d}"
    iris_stash_code = 'm01s'+temp[0:2]+'i'+temp[2:]
    return iris_stash_code


