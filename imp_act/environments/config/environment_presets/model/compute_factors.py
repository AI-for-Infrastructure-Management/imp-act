import numpy as np


# CAPACITY FACTORS
def get_capacity_factor(_days, _reduction):
    """Get the capacity factor for a given number of days and reduction.

    c * (356-_days) + c * _reduction * _days = c' * 365
    c'/c = ((356-_days) + _reduction * _days) / 365

    """
    total_days = 365

    return ((total_days - _days) + _reduction * _days) / total_days


num_actions = 5
capacity_factors = [0] * num_actions

# action 0: do-nothing
capacity_factors[0] = 1

# action 1: inspect
capacity_factors[1] = 1

# action 2: minor repair
_days = 4.5
_reduction = 0.25  # 25% reduction in capacity for 4.5 days
capacity_factors[2] = get_capacity_factor(_days, _reduction)

# action 3: major repair
_days = 8.5
_reduction = 0.5  # 50% reduction in capacity for 8.5 days
capacity_factors[3] = get_capacity_factor(_days, _reduction)

# action 4: replacement
_days = 42
_reduction = 0  # 100% reduction in capacity for 42 days
capacity_factors[4] = get_capacity_factor(_days, _reduction)


print(capacity_factors)


## Base Travel Time Factors
def get_btt_factor(_days, new_speed):
    """Get the BTT factor for a given number of days and reduction.

    - speed assumed is 80 km/h

    s * (356-_days) + s_new * _days  = s' * 365
    s/s' = 365 / ((356-_days) + (s_new/s) * _days)

    """
    total_days = 365

    _reduction = new_speed / 80

    return total_days / ((total_days - _days) + _reduction * _days)


btt_factors = [0] * num_actions

# action 0: do-nothing
btt_factors[0] = 1

# action 1: inspect
btt_factors[1] = 1

# action 2: minor repair
_days = 4.5
new_speed = 70  # 70 km/h for 4.5 days
btt_factors[2] = get_btt_factor(_days, new_speed)

# action 3: major repair
_days = 8.5
new_speed = 60  # 60 km/h for 8.5 days
btt_factors[3] = get_btt_factor(_days, new_speed)

# action 4: replacement
_days = 42
new_speed = 0  # 0 km/h for 42 days
btt_factors[4] = get_btt_factor(_days, new_speed)

print(btt_factors)
