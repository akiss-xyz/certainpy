#! /usr/bin/python3 -i

from math import atan, asin, sqrt, cos, sin, pi


## Flags
PROBABILISTIC_UNCERTAINTIES = True      # Assume that uncertainties are independent, random and 
                                        # normally distributed and hence in calculations make the
                                        # appropriate reductions, or if False, instead calculate 
                                        # maximum, determinate uncertainties.

FORMATTING_EXP = False                  # Print floats as either 0.0040 or 4e-3

# Used by manage_calls decorator, add your owns stuff there to hook into dunder operations
PRINT_CALL = False                      # Print calls to operators for debug purposes
PRINT_RESULT = False                    # Print output from functions for debug stuffs


## Constants
EPSILON = 0.000000001

## Decorators
def manage_calls(func):
    def wrapper(*args, **kwargs):
        if PRINT_CALL:
            print(func.__name__ + "(" + str(args) + "|" + str(kwargs) + ")")
        result = func(*args, **kwargs)
        if PRINT_RESULT:
            print(f"  = {result}")
        return result
    return wrapper

## Functions!
def partial(fun, i, values):
    nudged_values = [a for a in values]
    nudged_values[i] = values[i] + EPSILON
    if fun(nudged_values) == fun(values):
        # If we're at a point where fun isn't a function of the ith variable, we'll just nudge until
        # it is. This is an improvement over any other uncertainty calculator I found, for example,
        # the uncertainty of the operation z = x ** y where x = (1, +-0.2) y = (2, +-0.2) isn't
        # reported correctly by most such calculators. A better way to do this might be to look at
        # both sides of the input, and decide from there, but hey. That's for another day.
        # Another improvement that could be made is to find analytic solutions for partials, however
        # that's firstly beyond the scope of this project and also, evalfun and its friends were
        # intended to be the functions you turn to when nothing else works, so even if I implemented
        # these, they wouldn't live here.
        shifted_values = [a + EPSILON for a in nudged_values]
        return partial(fun, i, shifted_values)
    return (fun(nudged_values) - fun(values)) / EPSILON

def maximum_evaluate(fun, args):
    """ Evaluates the arbitrary function fun on the list of Uncertain_Value[s] args, with the
    assumption on fun that there are no turning points on the uncertain region.

    Params
    ------
    fun: a function of the form def your_function(inputs), where inputs is a list of floats. 
        For this strategy to work, you need to ensure that fun has no turning points anywhere
        within your uncertainties in args. Otherwise, your uncertainty value will be wrong.

    args: the list of Uncertain_Value[s] to evaluate at.

    Returns
    -------
    An Uncertain_Value with .val fun(args) and .unc corresponding to the uncertainty given by
    maximising the function and subtracting the best value of the function.

    See also
    --------
    partial(fun, i, values)
    evalfun(fun, values, no_turning=False)"""
    values = [a.val for a in args]
    maximised_args = []
    for i in range(len(args)):
        if partial(fun, i, values) > 0:
            maximised_args.append(args[i].max())
        else:
            maximised_args.append(args[i].min())
    return Uncertain_Value(fun(values), fun(maximised_args) - fun(values))

def prob_evalfun(fun, args):
    """
        Evaluates the arbitrary function fun on the list of Uncertain_Value[s] args, using the
        standard formula for propagating uncertainties in general functions. Specifically this is
        the probabilistic implementation, called by the general evalfun(fun, args)

        Params
        -----
        fun: a function of the form your_function(inputs), where inputs is a list of floats.

        args: the list of Uncertain_Value[s] to evaluate at.

        See also
        --------
        evalfun(fun, args, no_turning=False)
        nonprob_evalfun(fun, args, no_turning=False)
    """
    values = [a.val for a in args]
    sum_buffer = 0
    for i in range(len(args)):
        # estimate the partial with respect to arg at values
        partial_with_respect_to_i = partial(fun, i, values)
        #print(f"partial to {i}: {partial_with_respect_to_i}")
        sum_buffer = sum_buffer + (partial_with_respect_to_i * args[i].unc)**2

    unc_ans = sqrt(sum_buffer)
    return Uncertain_Value(fun(values), unc_ans)

def nonprob_evalfun(fun, args, no_turning=False):
    """
        Evaluates the arbitrary function fun on the list of Uncertain_Value[s] args, using the
        standard formula for propagating uncertainties in general functions. Specifically this is
        the non-probabilistic implementation, called by the general evalfun(fun, args)

        Params
        -----
        fun: a function of the form your_function(inputs), where inputs is a list of floats.

        args: the list of Uncertain_Value[s] to evaluate at.

        See also
        --------
        evalfun(fun, args, no_turning=False)
        prob_evalfun(fun, args, no_turning=False)
    """


    if no_turning:
        ## GOAL: make it check for turning points of fun on the uncertain region, and if there
        ## aren't any, then use maximum_evaluate
        return maximum_evaluate(fun, args)

    values = [a.val for a in args]
    sum_buffer = 0
    for i in range(len(args)):
        # estimate the partial with respect to arg at values
        partial_with_respect_to_i = abs(partial(fun, i, values))
        sum_buffer = sum_buffer + (partial_with_respect_to_i * args[i].unc)

    unc_ans = sum_buffer
    return Uncertain_Value(fun(values), unc_ans)

def evalfun(fun, args, no_turning=False):
    """ Evaluate a general function fun on the Uncertain_Value[s] in args. Deals automatically with
    whether you want PROBABILISTIC_UNCERTAINTIES, and dispatches accordingly to more specific
    functions.

    Params
        -----
        fun: a function of the form your_function(inputs), where inputs is a list of floats.

        args: the list of Uncertain_Value[s] to evaluate at.

        See also
        --------
        non_evalfun(fun, args, no_turning=False)
        prob_evalfun(fun, args, no_turning=False)
        maximum_evaluate(fun, args)
    """

    if no_turning and not PROBABILISTIC_UNCERTAINTIES:
        return maximum_evaluate(fun, args)

    if PROBABILISTIC_UNCERTAINTIES:
        return prob_evalfun(fun, args)
    else:
        return nonprob_evalfun(fun, args)

class Uncertain_Value:
    """ Represents a float with uncertainty. """

    def __init__(self, best_value, unc=0.0):
        self.unc = unc
        if unc < 0:
            self.unc = -unc
        self.val = best_value

    def fractional(self):
        """ Find the Uncertain_Value's fractional uncertainty. Positive definite. """
        if self.val < 0:
            return self.unc / (-1 * self.val)
        return self.unc / self.val

    def inverse_fractional(self, fractional):
        """ Inverse of the fractional function, used for calculating some uncertainties. """
        if self.val < 0:
            return fractional * (-1) * (self.val)
        return fractional * self.val

    def max(self):
        return self.val + self.unc

    def min(self):
        return self.val - self.unc

    @manage_calls
    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Uncertain_Value(self.val + other, self.unc)

        if isinstance(other, Uncertain_Value):
            if PROBABILISTIC_UNCERTAINTIES:
                return Uncertain_Value(self.val + other.val, sqrt(self.unc**2 + other.unc**2))
            else:
                return Uncertain_Value(self.val + other.val, self.unc + other.unc)

        raise TypeError(f"Uncertain_Value.__add__({self}, {other}) - __add__ not yet"
                " implemented for type of other. Try using evalfun and passing it the function.")

    __radd__ = __add__ 

    @manage_calls
    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return Uncertain_Value(self.val - other, self.unc)

        if isinstance(other, Uncertain_Value):
            if PROBABILISTIC_UNCERTAINTIES:
                return Uncertain_Value(self.val - other.val, sqrt(self.unc**2 + other.unc**2))
            else:
                return Uncertain_Value(self.val - other.val, self.unc + other.unc)

        raise TypeError(f"Uncertain_Value.__sub__({self}, {other}) - __sub__ not yet"
                " implemented for type of other. Try using evalfun and passing it the function.")

    @manage_calls
    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return Uncertain_Value(other- self.val, self.unc)

        if isinstance(other, Uncertain_Value):
            if PROBABILISTIC_UNCERTAINTIES:
                return Uncertain_Value(other.val - self.val, sqrt(self.unc**2 + other.unc**2))
            else:
                return Uncertain_Value(other.val - self.val, sqrt(self.unc + other.unc))

        raise TypeError(f"Uncertain_Value.__rsub__({self}, {other}) - __rsub__ not yet"
                " implemented for type of other. Try using evalfun and passing it the function.")

    @manage_calls
    def __mul__(self, other):
        if isinstance(other, (float, int)):
            answer = Uncertain_Value(self.val * other, 0.0)
            answer.unc = answer.inverse_fractional(self.fractional())
            return answer

        if isinstance(other, Uncertain_Value):
            if PROBABILISTIC_UNCERTAINTIES:
                answer = Uncertain_Value(other.val * self.val, 0.0)
                answer.unc = answer.inverse_fractional( sqrt(self.fractional()**2 + 
                    other.fractional()**2))
                return answer
            else:
                answer = Uncertain_Value(other.val * self.val, 0.0)
                answer.unc = answer.inverse_fractional( self.fractional() + other.fractional() 
                        + self.fractional()*other.fractional())
                return answer

        raise TypeError(f"Uncertain_Value.__mul__({self}, {other}) - __mul__ not yet"
                " implemented for type of other. Try using evalfun and passing it the function.")

    __rmul = __mul__

    @manage_calls
    def __div__(self, other):
        if isinstance(other, (float, int)):
            answer = Uncertain_Value(self.val / other, 0.0)
            answer.unc = answer.inverse_fractional(self.fractional())
            return answer

        if isinstance(other, Uncertain_Value):
            if PROBABILISTIC_UNCERTAINTIES:
                answer = Uncertain_Value(self.val / other.val, 0.0)
                answer.unc = answer.inverse_fractional(
                    sqrt(self.fractional()**2 + other.fractional()**2))
                return answer
            else:
                answer = Uncertain_Value(self.val / other.val, 0.0)
                answer.unc = answer.inverse_fractional(self.fractional() + other.fractional() 
                        + self.fractional()*other.fractional())
                return answer

        raise TypeError(f"Uncertain_Value.__div__({self}, {other}) - __div__ not yet"
                " implemented for type of other. Try using evalfun and passing it the function.")

    @manage_calls
    def __rdiv__(self, other):
        if isinstance(other, (float, int)):
            answer = Uncertain_Value(other / self.val, 0.0)
            answer.unc = answer.inverse_fractional(self.fractional())
            return answer

        if isinstance(other, Uncertain_Value):
            if PROBABILISTIC_UNCERTAINTIES:
                answer = Uncertain_Value(other.val / self.val, 0.0)
                answer.unc = answer.inverse_fractional(
                    sqrt(self.fractional()**2 + other.fractional()**2))
                return answer
            else:
                answer = Uncertain_Value(other.val / self.val, 0.0)
                answer.unc = answer.inverse_fractional(self.fractional() + other.fractional() 
                        + self.fractional()*other.fractional())
                return answer

        raise TypeError(f"Uncertain_Value.__div__({self}, {other}) - __div__ not yet"
                " implemented for type of other. Try using evalfun and passing it the function.")

    @manage_calls
    def __truediv__(self, other):
        if isinstance(other, (float, int)):
            answer = Uncertain_Value(self.val / other, 0.0)
            answer.unc = answer.inverse_fractional(self.fractional())
            return answer

        if isinstance(other, Uncertain_Value):
            if PROBABILISTIC_UNCERTAINTIES:
                answer = Uncertain_Value(self.val / other.val, 0.0)
                answer.unc = answer.inverse_fractional(
                    sqrt(self.fractional()**2 + other.fractional()**2))
                return answer
            else:
                answer = Uncertain_Value(self.val / other.val, 0.0)
                answer.unc = answer.inverse_fractional(self.fractional() + other.fractional() 
                        + self.fractional()*other.fractional())
                return answer

        raise TypeError(f"Uncertain_Value.__truediv__({self}, {other}) - __truediv__ not yet"
                " implemented for type of other. Try using evalfun and passing it the function.")

    @manage_calls
    def __pow__(self, other):
        def power(args):
            return args[0] ** args[1]

        if isinstance(other, (float, int)):
            return evalfun(power, [self, Uncertain_Value(other, 0.0)], no_turning=True)
                    # evalfun already checks if we're probabilistic

        if isinstance(other, Uncertain_Value):
            #print("pow of two uncertains")
            return evalfun(power, [self, other], no_turning=True)
                    # evalfun already checks if we're probabilistic

        try:
            a = other.val
            a = other.unc
        except:
            raise TypeError(f"Uncertain_Value.__pow__({self}, {other}) - __pow__ not yet"
                    " implemented for type of other. Try using evalfun and passing it the function."
                    " I tried it, but it seems like other didn't implement .val and .unc.")

        try:
            return evalfun(power, [self, other], no_turning=True)
        except:
            raise TypeError(f"Uncertain_Value.__pow__({self}, {other}) - __pow__ not yet"
                            " implemented for type of other. Try using evalfun and passing"
                            " it the function. I tried it, and while other had both .val and .unc,"
                            " evalfun threw an exception, so... Good luck?")

    def __repr__(self):
        if FORMATTING_EXP:
            return f"(" + f"{self.val:.3e}" + "+-" + f"{self.unc:.3e}" + f")"
        else:
            return f"(" + f"{self.val:.3f}" + "+-" + f"{self.unc:.3f}" + f")"

    def __lt__(self, other):
        """ Compare two Uncertain_Value instances, or a number and Uncertain_Value. 
        Currently is a simple bool, which isn't quite perfect, because if we compare two
        Uncertain_Values, there's often some probability that our comparison is true or false."""

        if isinstance(other, (int, float)):
            return self.val < other

        return self.val < other.val

    def __le__(self, other):
        """ Compare two Uncertain_Value instances, or a number and Uncertain_Value. 
        Currently is a simple bool, which isn't quite perfect, because if we compare two
        Uncertain_Values, there's often some probability that our comparison is true or false."""
        if isinstance(other, (int, float)):
            return self.val <= other

        if isinstance(other, Uncertain_Value):
            return self.val <= other.val

        raise TypeError(f"Uncertain_Value.__le__({self}, {other}) - __le__ not yet implemented"
                        " for type of other.")

    def __gt__(self, other):
        """ Compare two Uncertain_Value instances, or a number and Uncertain_Value. 
        Currently is a simple bool, which isn't quite perfect, because if we compare two
        Uncertain_Values, there's often some probability that our comparison is true or false."""
        if isinstance(other, (int, float)):
            return self.val > other

        if isinstance(other, Uncertain_Value):
            return self.val > other.val

        raise TypeError(f"Uncertain_Value.__gt__({self}, {other}) - __gt__ not yet implemented"
                        " for type of other.")

    def __ge__(self, other):
        """ Compare two Uncertain_Value instances, or a number and Uncertain_Value. 
        Currently is a simple bool, which isn't quite perfect, because if we compare two
        Uncertain_Values, there's often some probability that our comparison is true or false."""
        if isinstance(other, (int, float)):
            return self.val >= other

        if isinstance(other, Uncertain_Value):
            return self.val >= other.val

        raise TypeError(f"Uncertain_Value.__ge__({self}, {other}) - __ge__ not yet implemented"
                        " for type of other.")

class Vec:
    def __init__(self, x, y, mod_arg=False):
        """ A vector class that handles uncertainty. Params x and y should be Uncertain_Value
        instances. """

        self.x = x
        self.y = y
        self.mod_arg = mod_arg

    def __getitem__(self, key):
        if key in (0, 'x', 'r'):
            return self.x
        else:
            return self.y

    def __setitem__(self, key, value):
        if isinstance(value, (int, float)):
            if key in (0, 'x', 'r'):
                self.x.val = value
            else:
                self.y.val = value
            return self

        if isinstance(value, Uncertain_Value):
            if key in (0, 'x', 'r'):
                self.x = value
            else:
                self.y = value
            return self

        raise TypeError(f"Vec.__setitem__({self}, {value}) - __setitem__ not yet implemented"
                        "for type of value. Make sure you're assigning an Uncertain_Value.")

    @manage_calls
    def __pow__(self, other):
        """ Find the magnitude of the vector, returns an Uncertain_Value. """
        if self.mod_arg:
            return self[0]

        def magnitude(values):
            return sqrt(values[0]**2 + values[1]**2)

        if PROBABILISTIC_UNCERTAINTIES:
            # evalfun already checks if we're probabilistic, so this is okay
            return evalfun(magnitude, [self.x, self.y])
        else:
            # We're allowed to use maximum_evaluate because magnitude has no turning points on any
            # uncertain domain.
            return maximum_evaluate(magnitude, [self.x, self.y])

    @manage_calls
    def __pos__(self):
        """ Find the argument of the vector """
        if self.mod_arg:
            # We're already storing the argument as our second element
            return self[1]

        # TODO: what?

        # since atan x has no turning points, we can use maximum_evaluation on it.
        def inner_angle(values):
            return atan(values[1]/values[0])

        if PROBABILISTIC_UNCERTAINTIES:
            # Does this belong here? I mean this isn't very probabilistic, is it?
            if self.x == 0:
                if self.y > 0:
                    # On the positive y-axis
                    if not self.y.min() == 0:
                        return Uncertain_Value(pi / 2, atan(self.x.max() / self.y.min())) # TOTEST
                    else:
                        return Uncertain_Value(pi / 2, atan(self.x.max() / EPSILON))

                elif self.y < 0:
                    # On the negative y-axis
                    if not self.y.max() == 0:
                        return Uncertain_Value(-pi / 2, atan(self.x.min() / self.y.max())) # TOTEST
                    else:
                        return Uncertain_Value(-pi / 2, atan(self.x.min() / EPSILON))

                else:
                    # On the origin
                    return Uncertain_Value(0, 2 * pi) # TODO

            phi = evalfun(inner_angle, [self[0], self[1]])
            phi.val = abs(phi.val)

            if self.y >= 0 and self.x > 0:
                return phi
            if self.y >= 0 and self.x < 0:
                return pi - phi
            if self.y < 0 and self.x < 0:
                return phi - pi
            if self.y < 0 and self.x > 0:
                return 0-phi

            raise NotImplementedError(f"Vec.__pos__({self}): Evaluating with"
                            " PROBABILISTIC_UNCERTAINTIES, we slipped through the cracks somehow.")
        else:
            if self.x.val == 0:
                if self.y > 0:
                    # On the positive y-axis
                    if not self.y.min() == 0:
                        return Uncertain_Value(pi / 2, atan(self.x.max() / self.y.min())) # TOTEST
                    else:
                        return Uncertain_Value(pi / 2, atan(self.x.max() / EPSILON))

                elif self.y < 0:
                    # On the negative y-axis
                    if not self.y.max() == 0:
                        return Uncertain_Value(-pi / 2, atan(self.x.min() / self.y.max())) # TOTEST
                    else:
                        return Uncertain_Value(-pi / 2, atan(self.x.min() / EPSILON))

                else:
                    # On the origin
                    return Uncertain_Value(0, 2 * pi) # TODO

            # The inner angle of the vector with the x-axis
            phi = maximum_evaluate(inner_angle, [self[0], self[1]])
            phi.val = abs(phi.val)

            if self.y >= 0 and self.x > 0:
                return phi
            if self.y >= 0 and self.x < 0:
                return pi - phi
            if self.y < 0 and self.x < 0:
                return phi - pi
            if self.y < 0 and self.x > 0:
                return 0-phi

            raise NotImplementedError(f"Vec.__pos__({self}): Evaluating with no"
                    " PROBABILISTIC_UNCERTAINTIES, we slipped through the cracks somehow.")

    @manage_calls
    def __neg__(self):
        """ Returns an Uncertain_Value representing the magnitude of the vector. """
        return self**1

    @manage_calls
    def __add__(self, other):
        if isinstance(other, (int, float, Uncertain_Value)):
            if self.mod_arg:    # Let's take an add to a mod-arg as extending the length.
                return Vec(self[0] + other, self[1], mod_arg=True)
            else:               # An add to a Cartesian just moves the vector.
                return Vec(self[0] + other, self[1] + other)

        if isinstance(other, Vec):
            if self.mod_arg:
                if other.mod_arg:   # TODO: This is, uh, I'm not sure if this is gonna be good?
                    return ~(~other + ~self)
                else:               # Whoa this is a mess
                    return ~(other + ~self)
            else:
                if other.mod_arg:   # Please don't add vectors
                    return self + ~other
                else:               # This is fine
                    return Vec(self[0]+other[0], self[1]+other[1])

        raise TypeError(f"Vec.__neg__({self}, {other}) - __neg__ not yet implemented for type of other.")

    __radd__ = __add__

    @manage_calls
    def __sub__(self, other):
        if isinstance(other, (int, float, Uncertain_Value)):
            if self.mod_arg:    # Let's take an add to a mod-arg as extending the length.
                return Vec(self[0] - other, self[1], mod_arg=True)
            else:               # An add to a Cartesian just moves the vector.
                return Vec(self[0] - other, self[1] - other)

        if isinstance(other, Vec):
            if self.mod_arg:
                if other.mod_arg:   # TODO: Oooh mama.
                    return ~(~self - ~other)
                else:
                    return ~(~self - other)
            else:
                if other.mod_arg:
                    return self - ~other
                else:
                    return Vec(self[0]-other[0], self[1]-other[1])

        raise TypeError(f"Vec.__sub__({self}, {other}) - __sub__ not yet implemented for type of"
                        " other.")

    @manage_calls
    def __rsub__(self, other):
        if isinstance(other, (int, float, Uncertain_Value)):
            if self.mod_arg:    # Let's take an add to a mod-arg as extending the length.
                return Vec(other - self[0], self[1], mod_arg=True)
            else:               # An add to a Cartesian just moves the vector.
                return Vec(other - self[0], other - self[1])

        if isinstance(other, Vec):
            if self.mod_arg:
                if other.mod_arg:   # TODO: Oooh mama.
                    return ~(~other - ~self)
                else:
                    return ~(other - ~self)
            else:
                if other.mod_arg:
                    return ~other - self
                else:
                    return Vec(other[0] - self[0], other[1] - self[1])

        raise TypeError(f"Vec.__rsub__({self}, {other}) - __rsub__ not yet implemented for type"
                        " of other.")

    @manage_calls
    def __mul__(self, other):
        if isinstance(other, (int, float, Uncertain_Value)):
            if self.mod_arg:
                return Vec(self[0] * other, self[1], mod_arg=True)
            else:
                return Vec(self[0] * other, self[1] * other)

        if isinstance(other, Vec):
            if self.mod_arg:
                if other.mod_arg:
                    return Vec(self[0]*other[0], self[1]+other[1], mod_arg=True)
                else:
                    return ~(~self * other)
            else:
                if other.mod_arg:
                    return ~(self * ~other)
                else:
                    return Vec(v[0]*other[0], v[1]*other[1])

        raise TypeError(f"Vec.__mul__({self}, {other}) - __mul__ not yet implemented for type of"
                        " other.")

    __rmul__ = __mul__

    @manage_calls
    def __truediv__(self, other):
        if isinstance(other, (int, float, Uncertain_Value)):
            if self.mod_arg:
                return Vec(self[0] / other, self[1], mod_arg=True)
            else:
                return Vec(self[0] / other, self[1] / other)

        if isinstance(other, Vec):
            return Vec(self[0]/other[0], self[1]/other[1])

        raise TypeError(f"Vec.__truediv__({self}, {other}) - __truediv__ not yet implemented for"
                        " type of other.")

    @manage_calls
    def __div__(self, other):
        if isinstance(other, (int, float, Uncertain_Value)):
            if self.mod_arg:
                return Vec(self[0] / other, self[1], mod_arg=True)
            else:
                return Vec(self[0] / other, self[1] / other)

        if isinstance(other, Vec):
            return Vec(self[0]/other[0], self[1]/other[1])

        raise TypeError(f"Vec.__div__({self}, {other}) - __div__ not yet implemented for type"
                        " of other.")

    # eg ~v
    # used to go between mod_arg and cartesian

    @manage_calls
    def __invert__(self):
        """ Returns an inverted (e.g. if we're mod_arg, then it gives you it in Cartesian) version 
        of the vector, without touching the original """
        v = Vec(Uncertain_Value(0, 0), Uncertain_Value(0, 0))

        if self.mod_arg:
            if self[0].val == 0:
                # If our length is zero, then we have no clue about our x and y positions.
                v.x = Uncertain_Value(0, self[0].unc)
                v.y = Uncertain_Value(0, self[0].unc)
            else:
                # Since evalfun already checks for PROBABILISTIC_UNCERTAINTIES, we don't need to
                # here.
                r = self[0]

                def cosine(values):
                    return cos(values[0])

                def sine(values):
                    return sin(values[0])

                v.x = r * evalfun(cosine, [self[1]])
                v.y = r * evalfun(sine, [self[1]])

                v.mod_arg = False
        else:
            length = self**1
            v[1] = self.__pos__()
            v[0] = length

            v.mod_arg = True

        return v

    def inv(self):
        """ Returns an inverted (e.g. if we're mod_arg, then it gives you it in Cartesian) version 
        of the vector, and changes the original accordingly. """
        vec = ~self
        self[0] = vec[0]
        self[1] = vec[1]
        self.mod_arg = vec.mod_arg
        return self

    def __repr__(self):
        if self.mod_arg:
            return f"Vec({self.x}< {self.y})"
        return f"Vec({self.x}, {self.y})"

def empty_vec():
    return Vec(Uncertain_Value(0, 0), Uncertain_Value(0, 0))


if __name__ == "__main__":
    x = Uncertain_Value(1, 0.2)
    y = Uncertain_Value(2, 0.2)
    a = Uncertain_Value(1, 0.2)
    b = Uncertain_Value(1, 0.2)

    v = Vec(x, y)
    u = Vec(a, b)

    print(" --- Running test on:")
    print(f"  x = {x},  y = {y}")
    print(f"  v = {v},  u = {u}")

    print(" --- Testing Uncertain_Values:")
    print(" -- Calculating with PROBABILISTIC_UNCERTAINTIES=False")
    PROBABILISTIC_UNCERTAINTIES=False
    print(f"    OP    |   MAXIMAL VAL  | ANSWER")
    print(f" x  +  y  =     3+-0.4     = {x+y}")
    print(f" y  +  x  =     3+-0.4     = {y+x}")
    print(f" x  -  y  =    -1+-0.4     = {x-y}")
    print(f" y  -  x  =     1+-0.4     = {y-x}")
    print(f" x  *  y  =     2+-0.64    = {x*y}")
    print(f" y  *  x  =     2+-0.64    = {y*x}")
    print(f" x  /  y  =    .5+-0.16.   = {x/y}")
    print(f" y  /  x  =     2+-0.75    = {y/x}")
    print(f" x  ** y  =     1+-0.49.   = {x**y}")
    print(f" y  ** x  =     2+-0.58.   = {y**x}")

    print("")
    print(" -- Calculating with PROBABILISTIC_UNCERTAINTIES=True")
    PROBABILISTIC_UNCERTAINTIES=True
    print(f"    OP    |   PROBAB. VAL  |   MAXIMAL VAL  | ANSWER")
    print(f" x  +  y  =     3+-0.28.   =     3+-0.4     = {x+y}")
    print(f" y  +  x  =     3+-0.28.   =     3+-0.4     = {y+x}")
    print(f" x  -  y  =    -1+-0.28.   =    -1+-0.4     = {x-y}")
    print(f" y  -  x  =     1+-0.28.   =     1+-0.4     = {y-x}")
    print(f" x  *  y  =     2+-0.45.   =     2+-0.64    = {x*y}")
    print(f" y  *  x  =     2+-0.45.   =     2+-0.64    = {y*x}")
    print(f" x  /  y  =    .5+-0.11.   =    .5+-0.16.   = {x/y}")
    print(f" y  /  x  =     2+-0.45.   =     2+-0.75    = {y/x}")
    print(f" x  ** y  =     1+-0.40    =     1+-0.49.   = {x**y}")
    print(f" y  ** x  =     2+-0.34.   =     2+-0.58.   = {y**x}")

