from collections import namedtuple, OrderedDict


squeeze_channel = [
    'GAP',
    'L2-norm',
    'p-norm',
    'GAP+GMP',
    'GAP+std',
    'SkewPooling',
    '2nd-order pooling'
]

excitation_channel = [
    '3conv1D_3',
    '2conv_1x1',
    'conv1D_3',
    'conv1D_5',
    'conv1D_7',
    '2conv1D_3',
    'affine'
]

squeeze_spatial = [
    'GAP',
    'L2-norm',
    'p-norm',
    'GAP+GMP',
    'GAP+std',
    'SkewPooling',
    '2nd-order pooling'
]

excitation_spatial = [
    'conv_3x1_1x3',
    'conv_5x1_1x5',
    'conv_3x3',
    'conv_5x5',
    'conv_7x7',
    '2conv_3x3',
    'affine'
]


primitives_chsp = OrderedDict([('ch_sp', [squeeze_channel] + 2*[excitation_channel] + 2*[squeeze_spatial] + [excitation_spatial])])
