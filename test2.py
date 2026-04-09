import torch
from pprint import pprint
from numpy import array
from numpy import float32
from rich import print
print(
    {'three_expert': {'intro_batch_mean': array([0.46730167, 0.37100884, 0., 0.16168956, 0.,
                                                 0.], dtype=float32),
                      'inter_batch_var': array([6.6793080e-05, 6.2487437e-05, 0.0000000e+00, 1.0351351e-05,
                                                0.0000000e+00, 0.0000000e+00], dtype=float32),
                      'intro_batch_vars': array([0.06016639, 0.04937054, 0., 0.0266732, 0.,
                                                 0.], dtype=float32)},
     'ctr_contribute': {'intro_batch_mean': array([0.03968473, 1.2677209, 0., -0.901354, 0.,
                                                   0.], dtype=float32),
                        'inter_batch_var': array([0.00154491, 0.00162787, 0., 0.00041842, 0.,
                                                  0.], dtype=float32),
                        'intro_batch_vars': array([1.0264981, 2.8640242, 0., 1.1054627, 0., 0.],
                                                  dtype=float32)}},
)