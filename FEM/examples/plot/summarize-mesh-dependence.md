```vis
data:
  url: code/paralab/Dendro-KT/savedata/2023-07-11-anisotropy-cube.csv
transform:
  - filter: indexof(datum.solver, "accelerated") != -1
  - calculate: replace(datum.solver, "accelerated_", "cg+")
    as: solver
  - calculate: toString("Anisotropy(") + format(datum.scale_x, "d") + ":" + format(datum.scale_y, "d") + ":" + format(datum.scale_z, "d") + ")"
    as: stretch
  - calculate: format(datum.cells, ",d") + " (lev=" + toString(datum.max_depth) + ")"
    as: Cells (depth)
  - calculate: toNumber(datum.cells)
    as: cells
  - calculate: toNumber(datum.vcycles)
    as: vcycles
  - window:
    - op: first_value
      field: res_L2
      as: first_res_L2
    groupby: ["Cells (depth)", "solver", "stretch"]
  - calculate: toNumber(datum.res_L2) / toNumber(datum.first_res_L2)
    as: rel_res_L2
  - joinaggregate:
    - op: max
      field: vcycles
      as: max_vcycles
    groupby: ["Cells (depth)", "solver", "stretch"]
  - filter: "datum.vcycles == datum.max_vcycles"
  - calculate: -log(datum.rel_res_L2)/LN10 / datum.vcycles
    as: digits_per_vcycle
  - calculate: datum.vcycles * 10.0 / (-log(datum.rel_res_L2)/LN10)
    as: vcycles_per_10_digits
facet:
  field: stretch
  type: nominal
  header:
    title: false
    labelOrient: top
    labelPadding: -18
    labelAlign: right
    labelAnchor: end
#######    labelAnchor: middle
resolve:
  scale:
    y: independent
columns: 1
spacing: 5
spec:
  mark: line
  encoding:
    y:
      field: Cells (depth)
      type: nominal
      sort:
        field: cells
      title: Cells
    x:
      field: vcycles_per_10_digits
      type: quantitative
      scale:
        type: linear
    color:
      field: solver
      type: nominal
      legend: false
      scale:
        scheme: set1
    shape:
      field: solver
      type: nominal
      legend:
        orient: none
        direction: horizontal
        legendX: -60
        legendY: -30
        title: false
        columnPadding: 15

config:
  line:
    strokeWidth: 3
    point: true
  point:
    size: 75
  style:
    guide-label:
      font: Times
      fontSize: 12
    guide-title:
      font: Times
      fontSize: 14
      fontWeight: normal
    group-title:
      font: Times
```


