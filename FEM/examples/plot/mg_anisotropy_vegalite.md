
# Solver convergence on different meshes

```vis
data:
  url: Dendro-KT/savedata/anisotropic_vcycle_data.csv
params:
  - name: structure
    select: {type: point, fields: [mesh_family]}
    bind: {input: select, options: [adapt, uniform]}
transform:
  - filter: {param: structure}
  - calculate: format(datum.scale_x, "d") + "/" + format(datum.scale_y, "0.2f")
    as: scaling
  - calculate: format(datum.cells, ",d") + " (lev=" + toString(datum.max_depth) + ")"
    as: Cells (depth)
  - calculate: toNumber(datum.cells)
    as: cells
  - window:
    - op: first_value
      field: res_L2
      as: first_res_L2
    groupby: ["Cells (depth)", "solver"]
  - calculate: toNumber(datum.res_L2) / toNumber(datum.first_res_L2)
    as: rel_res_L2
facet:
  row:
    field: scaling
    type: nominal
spec:
  mark: line
  encoding:
    x:
      field: vcycles
      type: quantitative
    y:
      field: rel_res_L2
      type: quantitative
      scale:
        type: log
    column:
      field: Cells (depth)
      type: nominal
      sort:
        field: cells
        order: descending
    color:
      field: solver
      type: nominal
      legend:
        orient: "top"

config:
  line:
    strokeWidth: 3
  header:
    titleAngle: 0
    labelAngle: 0
```


# Mesh independence on different solvers

```vis
data:
  url: Dendro-KT/savedata/anisotropic_vcycle_data.csv
transform:
  - calculate: format(datum.cells, ",d") + " (lev=" + toString(datum.max_depth) + ")"
    as: Cells (depth)

  - calculate: toNumber(datum.cells)
    as: cells
  - window:
    - op: first_value
      field: res_L2
      as: first_res_L2
    groupby: ["Cells (depth)", "solver"]
  - calculate: toNumber(datum.res_L2) / toNumber(datum.first_res_L2)
    as: rel_res_L2

  - window:
    - op: lag
      field: rel_res_L2
      param: 2
      as: lag2_res_L2
    - op: lag
      field: rel_res_L2
      param: 1
      as: lag1_res_L2
    groupby: ["Cells (depth)", "solver"]

  - filter: (datum.vcycles <= 5) || (datum.rel_res_L2 < 0.90 * datum.lag2_res_L2)

  - joinaggregate:
      - op: min
        field: rel_res_L2
        as: min_res_L2
    groupby: ["Cells (depth)", "solver"]

  - joinaggregate:
      - op: max
        field: min_res_L2
        as: common_res_L2
    groupby: [solver]

  - filter: (datum.vcycles <= 5) || (datum.lag1_res_L2 >= datum.common_res_L2)

facet:
  column: {field: solver}
spec:
  vconcat:
    - mark: line
      encoding:
        x:
          field: vcycles
          type: quantitative
        y:
          field: rel_res_L2
          type: quantitative
          scale:
            type: log
          axis:
            format: "0.0e"
        color:
          field: "Cells (depth)"
          type: nominal
          sort: {field: cells, order: ascending}
    - mark: bar
      encoding:
        x:
          field: vcycles
          type: quantitative
          aggregate: max
        y:
          field: "Cells (depth)"
          type: nominal
          sort:
            field: vcycles
            op: max
  resolve:
    scale:
      x: shared
resolve:
  axis:
    y: shared

config:
  line:
    strokeWidth: 3
```
