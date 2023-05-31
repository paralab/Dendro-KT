import sys

if len(sys.argv) < 2:
    print("Usage: {} <data_file.hdf5>".format(sys.argv[0]), file=sys.stderr)
data_filename = sys.argv[1]

import h5py
import pandas as pd

# Input format (hdf5)
#     /
#      stored group name
#        attributes: ...
#        (datasets)
#          matvecs
#          vcycles
#          res_L2
#          res_Linf
#

# Output format (pandas -> csv)
#
#     group [attr1] [attr2] matvecs vcycles res_L2 res_Linf
#     ----- ------- ------- ------- ------- ------ --------
#      ..    ..      ..      ..      ..      ..     ..
#      ..    ..      ..      ..      ..      ..     ..
#

def flatten_dataset(group_name, group):
    # Put the adjacent datasets into columns.
    rows = pd.DataFrame({dataset_name: dataset for dataset_name, dataset in group.items()})
    # Broadcast name and attributes.
    rows['group_name'] = group_name
    for attr_key, attr_value in group.attrs.items():
        rows[attr_key] = attr_value
    return rows

with h5py.File(data_filename, 'r') as data:
    combined = pd.concat([flatten_dataset(group_name, group) for group_name, group in data.items()], ignore_index=True)
    # using csv because pandas DataFrame.to_json() is buggy w.r.t. precision.
    print(combined.to_csv(index=False), end="")

