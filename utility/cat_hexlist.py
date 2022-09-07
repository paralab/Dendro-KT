import sys
import json

if len(sys.argv) <= 1:
    print("Usage: python3 ", sys.argv[0], " file1 [file2 ...] > outfile")
    exit(1)

## Use metadata from first file.
py_data = json.load(open(sys.argv[1]))
format_version = py_data['format_version']

## Keep appending to the 'data' array.
for f in sys.argv[2:]:
    py_add = json.load(open(f));
    assert py_add['format_version'] == format_version, "Version mismatch between %s and %s" % (sys.argv[1], f)
    py_data['data'] += py_add['data']

## Back to compact json string. Caller can redirect output to file.
print(json.dumps(py_data, separators=(',', ':')), end='')
