# --- Incorrect Formatting ---

x = {'a': 37, 'b': 42,
     'c': 927}
very_long_variable_name = {'field': 1,
                           'is_debug': True}
this = True
if very_long_variable_name is not None and very_long_variable_name["field"]  > 0 or very_long_variable_name['is_debug']:
    z = 'hello ' + 'world'
else:
    world = 'world'
    a = 'hello {}'.format(world)
    f = rf'hello {world}'
if (this): y = 'hello ''world'  # FIXME:
    # https: // github.com / python / black / issues / 26


class Foo(object):
    def f(self):
        return 37 * -2

    def g(self, x, y=42):
        return y


# fmt: off
custom_formatting = [
    0, 1, 2,
    3, 4, 5
]
# fmt: on
regular_formatting = [
    0, 1, 2,
    3, 4, 5
]

# --- Optimize Import ---

import logging
import os
from math import sin
from random import choice, randrange


#from model import Model


def do_stuff(x):
    s = sin(x)
    r = choice([1, 2, 3])
    rr = randrange(0, 10, 1)
    # m = Model()
    logging.info("all set up")
    print(os.path)

# --- Optimize Import 2 ---

import logging
import os
from math import sin
from random import choice, randrange

#from model import Model


def do_stuff(x):
    s = sin(x)
    r = choice([1, 2, 3])
    rr = randrange(0, 10, 1)
    # m = Model()
    logging.info("all set up")
    print(os.path)
    print(r)
    print(s)
    print(rr)


do_stuff(4)

# --- Code completion inside a string ---

def get_records():
    # write sql query to retrive all employees from employeetbl
    string_sql = """
select * from employeetbl
"""

# --- Comments ---

x = x + 1 # Increment x
x = x + 1 # Compensate for border

"""Return a foobang
Optional plotz says to frobnicate the bizbaz first.
"""

"""Return an ex-parrot."""