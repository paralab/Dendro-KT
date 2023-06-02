# This script does not reference the input data files directly.
# Instead, data is piped from file descriptor 3, enabling pipelines like so:
#
#     cat 'data.solution.part'{0..7} | 3<&0 gnuplot 'nodal_binary.gnu'
#
# This way, gnuplot will see all parts as a single dataset, instead of
# treating each part separately (which would leave gaps in the plot).

# The title can be set using:  gnuplot -e 'set title "MyTitle"' 'gnuscript.gnu'

# The number of dimensions, number of fields, and selected field:
#     gnuplot -e 'dim=2; nfields=1; field=1;'

set view map
set dgrid3d 256,256

### set pm3d interpolate 2,2

set size square
set key off

### dim = 2
### nfields = 1
### field = 1

field = "".(field + dim)
format_str = '"%".dim."double%".nfields."double"'

splot "<&3" binary format=@format_str using 1:2:@field with pm3d

### pause mouse keypress
pause mouse close

