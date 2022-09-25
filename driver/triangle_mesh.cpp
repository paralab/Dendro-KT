
// ----------------------------------------------------------------------------
// Debugging
// ----------------------------------------------------------------------------
#include "external/suspect.h"


// ----------------------------------------------------------------------------
// Point
// ----------------------------------------------------------------------------

#include <stdint.h>
#include <array>

template <typename T, int dim>
struct Vec
{
  std::array<T, dim> data = {};

  constexpr bool operator==(const Vec &v) const { return data == v.data; }
  constexpr bool operator!=(const Vec &v) const { return data != v.data; }
  constexpr T operator[](int d) const { return data[d]; }
  T &operator[](int d) { return data[d]; }
};

template <typename T, int dim>
struct Point
{
  std::array<T, dim> data = {};

  constexpr T operator[](int d) const { return data[d]; }
  constexpr bool operator==(const Point &p) const { return data == p.data; }
  constexpr bool operator!=(const Point &p) const { return data != p.data; }
  T &operator[](int d) { return data[d]; }
};

using Vec2f = Vec<float, 2>;
using Vec3f = Vec<float, 3>;
using Point2f = Point<float, 2>;
using Point3f = Point<float, 3>;
using Point3u = Point<uint32_t, 3>;

#include <utility>

template <typename Expression, typename Seq>
struct ExprSeq {};
template <typename Expression, int...seq>
struct ExprSeq<Expression, std::integer_sequence<int, seq...>>
{
  template <typename Type>
  operator Type() const { return { expr(seq)... }; }
  Expression expr;
};

template <int dim, typename Expression>
auto dim_pack(Expression expr)
{
  return ExprSeq<Expression, std::make_integer_sequence<int, dim>>{expr};
}

template <typename T, int dim>
Vec<T, dim> operator-(Point<T, dim> a, Point<T, dim> b)
{
  return (Vec<T, dim>) dim_pack<dim>([&](int d){return (a[d] - b[d]);});
}

template <typename T, int dim>
Vec<T, dim> operator-(Vec<T, dim> a, Vec<T, dim> b)
{
  return (Vec<T, dim>) dim_pack<dim>([&](int d){return (a[d] - b[d]);});
}

template <typename T, int dim>
Vec<T, dim> operator+(Vec<T, dim> a, Vec<T, dim> b)
{
  return (Vec<T, dim>) dim_pack<dim>([&](int d){return (a[d] + b[d]);});
}

template <typename T, int dim>
Point<T, dim> operator+(Point<T, dim> a, Vec<T, dim> delta)
{
  return (Point<T, dim>) dim_pack<dim>([&](int d){return a[d] + delta[d];});
}

template <typename T, int dim>
Vec<T, dim> operator*(Vec<T, dim> v, T scale)
{
  return (Vec<T, dim>) dim_pack<dim>([&](int d){return v[d] * scale;});
}


template <typename T, int dim>
Point<T, dim-1> project_to_hyperplane(Point<T, dim> p, int axis)
{
  return (Point<T, dim-1>) dim_pack<dim-1>([&](int d){return p[d + (d >= axis)];});
}


template <typename T, int dim>
T dot(Vec<T, dim> a, Vec<T, dim> b)
{
  T sum = 0;
  for (int d = 0; d < dim; ++d)
    sum += a[d] * b[d];
  return sum;
}

template <typename T>
Vec<T, 2> perp(Vec<T, 2> a)
{
  return {-a[1], a[0]};
}

template <typename T>
Vec<T, 3> cross(Vec<T, 3> a, Vec<T, 3> b)
{
  return { a[1]*b[2]-b[1]*a[2],
          -a[0]*b[2]+b[0]*a[2],
           a[0]*b[1]-b[0]*a[1] };
}




#include <ostream>
namespace std
{
  ostream & operator<<(ostream &out, const Point3f &point)
  {
    out << "(" << point[0] << " " << point[1] << " " << point[2] << ")";
    return out;
  }

  ostream & operator<<(ostream &out, const Point3u &point)
  {
    out << "(" << point[0] << " " << point[1] << " " << point[2] << ")";
    return out;
  }
}

// ----------------------------------------------------------------------------
// Bounds
// ----------------------------------------------------------------------------

#include <limits>

struct Range
{
  // By default, empty range.
  float min = std::numeric_limits<float>::infinity();
  float max = -std::numeric_limits<float>::infinity();
};

struct Bounds
{
  Range ranges[3];
};

struct Frame
{
  Point3f origin;
  float twice_unit[3];
};


Range range_grow_factor(Range range, const float factor)
{
  const float margin = factor - 1;
  const float delta = 0.5f * margin * (range.max - range.min);
  range.min -= delta;
  range.max += delta;
  return range;
}

void range_include_point(Range &range, float point)
{
  if (point < range.min)
    range.min = point;
  if (point > range.max)
    range.max = point;
}

bool range_contains_point(const Range range, const float point)
{
  return range.min <= point and point <= range.max;
}

bool range_disjoint(const Range a, const Range b)
{
  return a.max < b.min or b.max < a.min;
}

float range_midpoint(const Range &range)
{
  return 0.5f * (range.min + range.max);
}

float range_length(const Range &range)
{
  return range.max - range.min;
}

Range range_set_length(Range range, const float length)
{
  const float midpoint = range_midpoint(range);
  range.min = midpoint - length * 0.5f;
  range.max = midpoint + length * 0.5f;
  return range;
}

Bounds bounds_grow_factor(Bounds bounds, float factor)
{
  for (int d = 0; d < 3; ++d)
    bounds.ranges[d] = range_grow_factor(bounds.ranges[d], factor);
  return bounds;
}

void bounds_include_point(Bounds &bounds, const Point3f point)
{
  for (int d = 0; d < 3; ++d)
    range_include_point(bounds.ranges[d], point[d]);
}

bool bounds_contains_point(const Bounds bounds, const Point3f point)
{
  return range_contains_point(bounds.ranges[0], point[0]) and
         range_contains_point(bounds.ranges[1], point[1]) and
         range_contains_point(bounds.ranges[2], point[2]);
}

bool bounds_disjoint(const Bounds a, const Bounds b)
{
  for (int d = 0; d < 3; ++d)
    if (range_disjoint(a.ranges[d], b.ranges[d]))
      return true;
  return false;
}

Bounds bounds_cube(Bounds bounds)
{
  float max_length = range_length(bounds.ranges[0]);
  for (int d = 1; d < 3; ++d)
  {
    const float length = range_length(bounds.ranges[d]);
    if (length > max_length)
      max_length = length;
  }
  for (int d = 0; d < 3; ++d)
    /// bounds.ranges[d] = range_set_length(bounds.ranges[d], max_length);
    range_include_point(bounds.ranges[d], bounds.ranges[d].min + max_length);
  return bounds;
}

Frame bounds_frame(const Bounds &bounds)
{
  Frame frame;
  for (int d = 0; d < 3; ++d)
    frame.origin.data[d] = range_midpoint(bounds.ranges[d]);
  for (int d = 0; d < 3; ++d)
    frame.twice_unit[d] = range_length(bounds.ranges[d]);
  return frame;
}

#include <ostream>
namespace std
{
  ostream & operator<<(ostream &out, const Bounds &bounds)
  {
    out << "[" << bounds.ranges[0].min << " " << bounds.ranges[0].max << "]"
        << " : "
        << "[" << bounds.ranges[1].min << " " << bounds.ranges[1].max << "]"
        << " : "
        << "[" << bounds.ranges[2].min << " " << bounds.ranges[2].max << "]";
    return out;
  }
}

// ----------------------------------------------------------------------------
// stl_trimesh
// ----------------------------------------------------------------------------

#include <istream>
#include <ostream>

struct stl_frontmatter
{
  uint8_t  header[80] = {};
  uint32_t n_triangles = 0;
};

struct stl_tri
{
  Vec3f    normal = {};
  Point3f  vertex[3] = {};
  uint16_t attribute_byte_count = 0;
};

struct stl_trimesh
{
  stl_frontmatter frontmatter = {};
  stl_tri * triangles = nullptr;
};

stl_frontmatter stl_frontmatter_create(uint32_t n_triangles)
{
  stl_frontmatter frontmatter;
  frontmatter.n_triangles = n_triangles;
  return frontmatter;
}

stl_frontmatter stl_frontmatter_create_from_file(std::istream &in)
{
  stl_frontmatter frontmatter;
  in.read((std::istream::char_type*) frontmatter.header, sizeof frontmatter.header);
  in.read((std::istream::char_type*) &frontmatter.n_triangles, sizeof frontmatter.n_triangles);
  return frontmatter;
}

void stl_frontmatter_to_file(std::ostream &out, const stl_frontmatter &frontmatter)
{
  out.write((std::ostream::char_type*) frontmatter.header, sizeof frontmatter.header);
  out.write((std::ostream::char_type*) &frontmatter.n_triangles, sizeof frontmatter.n_triangles);
}

stl_tri stl_tri_create_from_file(std::istream &in)
{
  stl_tri tri;
  in.read((std::istream::char_type*) &tri.normal, sizeof tri.normal);
  in.read((std::istream::char_type*) tri.vertex, sizeof tri.vertex);
  in.read((std::istream::char_type*) &tri.attribute_byte_count, sizeof tri.attribute_byte_count);
  return tri;
}

void stl_tri_to_file(std::ostream &out, const stl_tri &tri)
{
  out.write((std::ostream::char_type*) &tri.normal, sizeof tri.normal);
  out.write((std::ostream::char_type*) tri.vertex, sizeof tri.vertex);
  out.write((std::ostream::char_type*) &tri.attribute_byte_count, sizeof tri.attribute_byte_count);
}

stl_trimesh stl_trimesh_create(stl_frontmatter frontmatter)
{
  stl_trimesh trimesh;
  trimesh.frontmatter = frontmatter;
  const auto n_triangles = trimesh.frontmatter.n_triangles;
  trimesh.triangles = new stl_tri[n_triangles];
  return trimesh;
}

void stl_trimesh_set_tri(stl_trimesh &trimesh, int i, stl_tri tri)
{
  trimesh.triangles[i] = tri;
}

stl_tri stl_trimesh_get_tri(const stl_trimesh &trimesh, int i)
{
  return trimesh.triangles[i];
}

stl_trimesh stl_trimesh_create_from_file(std::istream &in)
{
  stl_trimesh trimesh = stl_trimesh_create(stl_frontmatter_create_from_file(in));
  const auto n_triangles = trimesh.frontmatter.n_triangles;
  for (int i = 0; i < n_triangles; ++i)
    stl_trimesh_set_tri(trimesh, i, stl_tri_create_from_file(in));
  return trimesh;
}


void stl_trimesh_to_file(std::ostream &out, const stl_trimesh &trimesh)
{
  const auto n_triangles = trimesh.frontmatter.n_triangles;
  stl_frontmatter_to_file(out, trimesh.frontmatter);
  for (int i = 0; i < n_triangles; ++i)
    stl_tri_to_file(out, trimesh.triangles[i]);
}

void stl_trimesh_destroy(stl_trimesh &trimesh)
{
  delete [] trimesh.triangles;
  trimesh.triangles = nullptr;
}

Bounds stl_trimesh_bounds(const stl_trimesh &trimesh)
{
  Bounds bounds = {};
  const auto n_triangles = trimesh.frontmatter.n_triangles;
  const stl_tri *triangle = trimesh.triangles;
  for (auto triangles_end = triangle + n_triangles; triangle != triangles_end; ++triangle)
    for (int v = 0; v < 3; ++v)
      bounds_include_point(bounds, triangle->vertex[v]);
  return bounds;
}


// ----------------------------------------------------------------------------
// octree
// ----------------------------------------------------------------------------

#include <assert.h>

Point3f frame_point_global_to_local(Point3f point, const Frame &frame)
{
  for (int d = 0; d < 3; ++d)
    point.data[d] = (point.data[d] - frame.origin.data[d]) / frame.twice_unit[d] + 0.5f;
  return point;
}

Point3f frame_point_local_to_global(Point3f point, const Frame &frame)
{
  for (int d = 0; d < 3; ++d)
    point.data[d] = (point.data[d] - 0.5f) * frame.twice_unit[d] + frame.origin.data[d];
  return point;
}

Bounds frame_bounds_local_to_global(Bounds bounds, const Frame &frame)
{
  for (int d = 0; d < 3; ++d)
  {
    bounds.ranges[d].min = (bounds.ranges[d].min - 0.5f) * frame.twice_unit[d] + frame.origin.data[d];
    bounds.ranges[d].max = (bounds.ranges[d].max - 0.5f) * frame.twice_unit[d] + frame.origin.data[d];
  }
  return bounds;
}

Point3u point_float_to_fixed(Point3f point, const int max_depth)
{
  const auto int_scale = 1u << max_depth;
  Point3u fixed;
  for (int d = 0; d < 3; ++d)
  {
    assert(0 <= point.data[d]);
    assert(point.data[d] < 1);
    fixed.data[d] = point.data[d] * int_scale;
  }
  return fixed;
}

Point3f point_fixed_to_float(Point3u fixed, const int max_depth)
{
  const auto int_scale = 1u << max_depth;
  Point3f p_float;
  for (int d = 0; d < 3; ++d)
  {
    p_float.data[d] = 1.0 * fixed.data[d] / int_scale;
  }
  return p_float;
}


#include <treeNode.h>
#include <tsort.h>

int max_depth() { return m_uiMaxDepth; }

constexpr int DIM = 3;

struct Octant
{
  ot::TreeNode<uint32_t, DIM> tree_node;
};

struct Octree
{
  std::vector<ot::TreeNode<uint32_t, DIM>> tree_nodes;
};

Octree octree_create_base(void)
{
  return Octree{{ot::TreeNode<uint32_t, DIM>()}};
}

Point3u tree_coord_to_fixed(periodic::PCoord<uint32_t, DIM> coord)
{
  return {{coord.coord(0), coord.coord(1), coord.coord(2)}};
}

Point3f tree_coord_to_local(periodic::PCoord<uint32_t, DIM> coord)
{
  return point_fixed_to_float(tree_coord_to_fixed(coord), max_depth());
}

Bounds tree_node_to_local_bounds(ot::TreeNode<uint32_t, DIM> octant)
{
  Bounds bounds;
  Point3u corner = tree_coord_to_fixed(octant.range().min());
  bounds_include_point(bounds, point_fixed_to_float(corner, max_depth()));
  for (int d = 0; d < 3; ++d)
    corner.data[d] += octant.range().side();
  bounds_include_point(bounds, point_fixed_to_float(corner, max_depth()));
  return bounds;
}




// ----------------------------------------------------------------------------
// possible Dendro-KT interface
// ----------------------------------------------------------------------------

// bool is_intercepted(Bounds box, int level, const Data *data, size_t begin, size_t end);
// bool is_interior(Bounds box, int level, const Data *data, size_t begin, size_t end, Point3d point);
// bool do_subdivide(Bounds box, int level, const Data *data, size_t begin, size_t end);


// ----------------------------------------------------------------------------
// octree to stl
// ----------------------------------------------------------------------------

#include <array>

stl_trimesh octree_to_trimesh(const Octree &octree, const Frame frame, bool wireframe_mode)
{
  const auto n_octants = octree.tree_nodes.size();
  const auto n_faces = n_octants * 6;
  const auto n_triangles = n_faces * 2;

  stl_trimesh trimesh = stl_trimesh_create(stl_frontmatter_create(n_triangles));

  int triangle_idx = 0;
  const auto emit = [&trimesh, &triangle_idx](const stl_tri &tri) {
    stl_trimesh_set_tri(trimesh, triangle_idx++, tri);
  };

  for (const ot::TreeNode<uint32_t, DIM> &tn : octree.tree_nodes)
  {
    // Get octant properties.
    const auto side = tn.range().side();
    const auto anchor = tn.getX();
    const Point3u anchor_fixed = {{anchor[0], anchor[1], anchor[2]}};

    // Compute octant vertices.
    Point3f vert[8];
    for (uint32_t v = 0; v < 8; ++v)
    {
      Point3u vertex_fixed = anchor_fixed;
      for (int d = 0; d < 3; ++d)
        if ((v & (1u << d)) != 0)
          vertex_fixed.data[d] += side;
      vert[v] = frame_point_local_to_global(point_fixed_to_float(vertex_fixed, max_depth()), frame);
    }

    const auto make_tri = [&vert](std::array<int, 3> v, int n) -> stl_tri
    {
      stl_tri tri;
      tri.normal.data[n / 2] = 1.0 - 2.0 * (n % 2);
      tri.vertex[0] = vert[v[0]];
      tri.vertex[1] = vert[v[1]];
      tri.vertex[2] = vert[v[2]];
      return tri;
    };

    if (wireframe_mode)
    {
      // Emit degenerate triangles for all edges.
      // -X to +X
      emit(make_tri({0,0,1}, 0));
      emit(make_tri({2,2,3}, 0));
      emit(make_tri({4,4,5}, 0));
      emit(make_tri({6,6,7}, 0));
      // -Y to +Y
      emit(make_tri({0,0,2}, 1));
      emit(make_tri({1,1,3}, 1));
      emit(make_tri({4,4,6}, 1));
      emit(make_tri({5,5,7}, 1));
      // -Z to +Z
      emit(make_tri({0,0,4}, 2));
      emit(make_tri({1,1,5}, 2));
      emit(make_tri({2,2,6}, 2));
      emit(make_tri({3,3,7}, 2));
    }
    else
    {
      // Emit triangles (with right-hand-rule) over octant surface.
      // Touching vertex 0
      emit(make_tri({0, 1, 5}, 2));
      emit(make_tri({0, 5, 4}, 2));
      emit(make_tri({0, 4, 6}, 0));
      emit(make_tri({0, 6, 2}, 0));
      emit(make_tri({0, 2, 3}, 4));
      emit(make_tri({0, 3, 1}, 4));
      // Touching vertex 7
      emit(make_tri({5, 1, 7}, 1));
      emit(make_tri({4, 5, 7}, 5));
      emit(make_tri({6, 4, 7}, 5));
      emit(make_tri({2, 6, 7}, 3));
      emit(make_tri({3, 2, 7}, 3));
      emit(make_tri({1, 3, 7}, 1));
    }
  }

  return trimesh;
}



// ----------------------------------------------------------------------------
// Triangle intersector
// ----------------------------------------------------------------------------

struct Triangle2D
{
  Point2f vertex[3] = {};
};

struct Triangle
{
  Point3f vertex[3] = {};
};

Triangle2D project_to_hyperplane(Triangle tri, int axis)
{
  return {{project_to_hyperplane(tri.vertex[0], axis),
           project_to_hyperplane(tri.vertex[1], axis),
           project_to_hyperplane(tri.vertex[2], axis)}};
}



std::vector<Triangle> stl_extract_triangles(const stl_trimesh &trimesh)
{
  const size_t n_triangles = trimesh.frontmatter.n_triangles;

  std::vector<Triangle> triangles;
  triangles.reserve(n_triangles);
  for (size_t i = 0; i < n_triangles; ++i)
  {
    stl_tri tri = stl_trimesh_get_tri(trimesh, i);
    triangles.push_back(Triangle{tri.vertex[0], tri.vertex[1], tri.vertex[2]});
  }
  return triangles;
}


struct LogTriIntercept;


struct PointInTriangle
{
  Point2f mid[3];
  Vec2f eperp[3];
  float dot_vert[3];

  PointInTriangle(Triangle2D tri)
  {
    struct VE { int edge_v0; int edge_v1; int vertex; };
    for (VE ve : { VE{0, 1, 2},  VE{1, 2, 0},  VE{2, 0, 1} })
    {
      Vec2f edge = tri.vertex[ve.edge_v1] - tri.vertex[ve.edge_v0];

      // Midpoint and perpendicular vector of triangle side opposite each vertex
      mid[ve.vertex] = tri.vertex[ve.edge_v0] + (edge * 0.5f);
      eperp[ve.vertex] = perp(edge);

      // If the query point is not in the same half-space as the opposite vertex,
      // it is not in the triangle.
      dot_vert[ve.vertex] = dot(tri.vertex[ve.vertex] - mid[ve.vertex], eperp[ve.vertex]);
    }
  }

  bool test_point(Point2f p) const
  {
    // If the query point is not in the same half-space as the opposite vertex,
    // it is not in the triangle.
    for (int v = 0; v < 3; ++v)
    {
      const float dot_p = dot(p - mid[v], eperp[v]);
      if (dot_p < 0.0f and dot_vert[v] >= 0.0f or
          dot_p > 0.0f and dot_vert[v] <= 0.0f)
        return false;
    }
    return true;
  }
};


bool intercepts(Triangle tri, Bounds bounds)
{
  SUSPECT_STATIC_COUNTER(int, enter_test, LogTriIntercept);

  Bounds tri_bounds;
  bounds_include_point(tri_bounds, tri.vertex[0]);
  bounds_include_point(tri_bounds, tri.vertex[1]);
  bounds_include_point(tri_bounds, tri.vertex[2]);
  if (bounds_disjoint(tri_bounds, bounds))
    return false;

  SUSPECT_STATIC_COUNTER(int, bounds_overlap, LogTriIntercept);

  const bool contains_vertex = bounds_contains_point(bounds, tri.vertex[0]) or
                               bounds_contains_point(bounds, tri.vertex[1]) or
                               bounds_contains_point(bounds, tri.vertex[2]);
  if (contains_vertex)
    return true;
  else
  {
    SUSPECT_STATIC_COUNTER(int, no_vertex, LogTriIntercept);

    // Otherwise, triangle intercepts bounds iff there is an edge of the cube
    //   that intersects the triangle plane in the region of the triangle

    const Vec3f e01 = tri.vertex[1] - tri.vertex[0];
    const Vec3f e02 = tri.vertex[2] - tri.vertex[0];
    const Vec3f tri_out_perp = cross(e01, e02);

    const auto box_vert = [&bounds](uint32_t i) -> Point3f { return
        { (i & 1u)? bounds.ranges[0].min : bounds.ranges[0].max,
          (i & 2u)? bounds.ranges[1].min : bounds.ranges[1].max,
          (i & 4u)? bounds.ranges[2].min : bounds.ranges[2].max }; };

    // Compute on which side of the triangle plane is each vertex of the box.
    enum TestHalfSpace : char { inside = 0, boundary = 1, outside = 2 };
    TestHalfSpace box_vert_test[8];
    for (int i = 0; i < 8; ++i)
    {
      const float sign = dot(box_vert(i) - tri.vertex[0], tri_out_perp);
                                        //(inside)   //boundary    //outside
      box_vert_test[i] = static_cast<TestHalfSpace>((sign >= 0) + (sign > 0));
    }

    const auto same_sign = [&box_vert_test](int i, int j) {
      return box_vert_test[i] == box_vert_test[j];
    };

    {const int axis = 0;
      const PointInTriangle triangle_region = {project_to_hyperplane(tri, axis)};
      for (int src : {0, 2, 4, 6})
      {
        const int dst = src + (1<<axis);
        if (not same_sign(src, dst) & triangle_region.test_point(
              project_to_hyperplane(box_vert(src), axis)))
          return true;
      }
    }
    {const int axis = 1;
      const PointInTriangle triangle_region = {project_to_hyperplane(tri, axis)};
      for (int src : {0, 1, 4, 5})
      {
        const int dst = src + (1<<axis);
        if (not same_sign(src, dst) & triangle_region.test_point(
              project_to_hyperplane(box_vert(src), axis)))
          return true;
      }
    }
    {const int axis = 2;
      const PointInTriangle triangle_region = {project_to_hyperplane(tri, axis)};
      for (int src : {0, 1, 2, 3})
      {
        const int dst = src + (1<<axis);
        if (not same_sign(src, dst) & triangle_region.test_point(
              project_to_hyperplane(box_vert(src), axis)))
          return true;
      }
    }

    SUSPECT_STATIC_COUNTER(int, not_cuts_edge, LogTriIntercept);
    return false;
  }



}

bool is_interior(const Triangle *begin, const Triangle *end, Point3f point)
{
  //TODO
  return false;
}

bool is_interior(const Triangle *begin, const Triangle *end, std::array<uint32_t, DIM> coord, Frame frame)
{
  const Point3u centroid_fixed = {{coord[0], coord[1], coord[2]}};
  const Point3f centroid = frame_point_local_to_global( point_fixed_to_float(
        centroid_fixed, max_depth()), frame);
  return is_interior(begin, end, centroid);
}

template <typename T, int n>
class count_distinct_small
{
  public:
    // Returns up to n for exact cardinality or n+1 if greater than n.
    int count() const {
      return seen;
    }
    void observe(const T &value) {
      if (seen <= n) {
        int i = 0;
        while (i < seen and values[i] != value)
          ++i;
        if (not (i < seen))
        {
          ++seen;
          if (i < n)
            values[i] = value;
        }
      }
    }
  private:
    std::array<T, n> values;
    int seen = 0;
};

bool do_refine_base(ot::TreeNode<uint32_t, DIM> octant, Bounds bounds,
    const Triangle *data, size_t begin, size_t end)
{
  if (octant.getLevel() == max_depth())
    return false;

  const size_t n_tris = end - begin;
  if (n_tris > 100 and octant.getLevel() < 10)
    return true;

  constexpr int vertex_threshold = 1;
  count_distinct_small<Point3f, vertex_threshold> count_verts;
  for (size_t i = begin; (i < end) & (count_verts.count() <= vertex_threshold); ++i)
  {
    Triangle tri = data[i];
    for (int v = 0; v < 3; ++v)
      if (bounds_contains_point(bounds, tri.vertex[v]))
        count_verts.observe(tri.vertex[v]);
  }
  return count_verts.count() > vertex_threshold;
}

bool do_refine_sub(Bounds box, int level, const Triangle *data, size_t begin, size_t end)
{
  //TODO
  return level < 4;
}

//
// Recursive refinement
//

class FrontStackIterator;
class BackStackIterator;

template <typename T>
class FrontStack
{
  public:
    FrontStack() = default;
    FrontStack(std::vector<T> &&array);
    ~FrontStack() = default;

    size_t size() const;
    size_t capacity() const;
    void resize(size_t size);
    void reserve(size_t capacity);

    template <typename InputIt>
    void insert_front(InputIt first, InputIt last);
    void push_front(T value);
    void pop_front(size_t count = 1);
    const T & peek_front() const;

    using Iterator = FrontStackIterator;
    Iterator begin() const;
    Iterator end() const;
    const T & operator[](Iterator it) const;
    T & operator[](Iterator it);

  private:
    std::vector<T> m_array;
    size_t m_size = 0;
};

template <typename T>
class BackStack
{
  public:
    BackStack() = default;
    BackStack(std::vector<T> &&array);
    ~BackStack() = default;

    size_t size() const;
    size_t capacity() const;
    void resize(size_t size);
    void reserve(size_t capacity);

    template <typename InputIt>
    void insert_back(InputIt first, InputIt last);
    void push_back(T value);
    void pop_back(size_t count = 1);
    const T & peek_back() const;

    using Iterator = BackStackIterator;
    Iterator begin() const;
    Iterator end() const;
    const T & operator[](Iterator it) const;
    T & operator[](Iterator it);

    std::vector<T> to_vector() &&;

  private:
    std::vector<T> m_array;
};

class FrontStackIterator
{
  public:
    FrontStackIterator operator+(size_t delta) const  { return {to_end - delta}; }
    FrontStackIterator operator-(size_t delta) const  { return {to_end + delta}; }
    size_t operator-(FrontStackIterator y) const      { return y.to_end - to_end; }
  private:
    template <typename T> friend class FrontStack;
    FrontStackIterator(size_t distance_to_end) : to_end(distance_to_end) {}
    size_t to_end = 0;
};

class BackStackIterator
{
  public:
    BackStackIterator operator+(size_t delta) const { return {idx + delta}; }
    BackStackIterator operator-(size_t delta) const { return {idx - delta}; }
    size_t operator-(BackStackIterator y) const     { return idx - y.idx; }
  private:
    template <typename T> friend class BackStack;
    BackStackIterator(size_t idx) : idx(idx) {}
    size_t idx = 0;
};


//
// FrontStack member functions
//

// This implementation only works for types that are default-constructible
template <typename T>
FrontStack<T>::FrontStack(std::vector<T> &&array) : m_array(std::move(array)), m_size(m_array.size())
{
}

template <typename T>
size_t FrontStack<T>::size() const
{
  return m_size;
}

template <typename T>
size_t FrontStack<T>::capacity() const
{
  return m_array.size();
}

template <typename T>
void FrontStack<T>::resize(size_t size)
{
  reserve(size);
  m_size = size;
}

template <typename T>
void FrontStack<T>::reserve(size_t capacity)
{
  if (capacity > this->capacity())
  {
    // Invariant: end of data coincides with end of internal vector.
    size_t end = m_array.size();
    size_t begin = end - m_size;
    m_array.reserve(capacity);
    m_array.resize(m_array.capacity());
    std::move_backward(m_array.begin() + begin, m_array.begin() + end, m_array.end());
  }
}

template <typename T>
FrontStackIterator FrontStack<T>::begin() const
{
  return {m_size};
}

template <typename T>
FrontStackIterator FrontStack<T>::end() const
{
  return {0};
}

template <typename T>
const T &FrontStack<T>::operator[](FrontStackIterator it) const
{
  return *(m_array.end() - it.to_end);
}

template <typename T>
T &FrontStack<T>::operator[](FrontStackIterator it)
{
  return *(m_array.end() - it.to_end);
}

template <typename T>
template <typename InputIt>
void FrontStack<T>::insert_front(InputIt first, InputIt last)
{
  this->resize(this->size() + (last - first));
  std::copy(first, last, &(*this)[this->begin()]);
}

template <typename T>
void FrontStack<T>::push_front(T value)
{
  this->resize(this->size() + 1);
  (*this)[this->begin()] = value;
}

template <typename T>
void FrontStack<T>::pop_front(size_t count)
{
  this->resize(this->size() - count);
}

template <typename T>
const T &FrontStack<T>::peek_front() const
{
  return (*this)[this->begin()];
}



//
// BackStack member functions
//


template <typename T>
BackStack<T>::BackStack(std::vector<T> &&array) : m_array(std::move(array))
{
}

template <typename T>
size_t BackStack<T>::size() const
{
  return m_array.size();
}

template <typename T>
size_t BackStack<T>::capacity() const
{
  return m_array.capacity();
}

template <typename T>
void BackStack<T>::resize(size_t size)
{
  m_array.resize(size);
}

template <typename T>
void BackStack<T>::reserve(size_t capacity)
{
  m_array.reserve(capacity);
}

template <typename T>
template <typename InputIt>
void BackStack<T>::insert_back(InputIt first, InputIt last)
{
  m_array.insert(m_array.end(), first, last);
}

template <typename T>
void BackStack<T>::push_back(T value)
{
  m_array.push_back(std::move(value));
}

template <typename T>
void BackStack<T>::pop_back(size_t count)
{
  m_array.resize(m_array.size() - count);
}

template <typename T>
const T &BackStack<T>::peek_back() const
{
  return m_array.back();
}

template <typename T>
BackStackIterator BackStack<T>::begin() const
{
  return {0};
}

template <typename T>
BackStackIterator BackStack<T>::end() const
{
  return {this->size()};
}

template <typename T>
const T &BackStack<T>::operator[](BackStackIterator it) const
{
  return m_array[it.idx];
}

template <typename T>
T &BackStack<T>::operator[](BackStackIterator it)
{
  return m_array[it.idx];
}

template <typename T>
std::vector<T> BackStack<T>::to_vector() &&
{
  return std::move(m_array);
}


void triangle_tree_sub(
    const Triangle *begin,
    const Triangle *end,
    std::vector<ot::TreeNode<uint32_t, DIM>> &sub_tier,
    ot::TreeNode<uint32_t, DIM> octant,
    ot::SFC_State<DIM> sfc)
{
  //TODO
}


void triangle_tree(
    FrontStack<Triangle> &input,
    std::array<FrontStackIterator, 2> range,
    BackStack<Triangle> &output,
    std::vector<ot::TreeNode<uint32_t, DIM>> &base_tier,
    std::vector<ot::TreeNode<uint32_t, DIM>> &sub_tier,
    Frame frame,
    ot::TreeNode<uint32_t, DIM> octant,
    ot::SFC_State<DIM> sfc)
{
  //           ______     ______
  // BackStack |_____ <-- _____|  FrontStack
  //

  // input: front segment in a FrontStack
  // pre: replace front segment with some subsets (may grow/shrink the segment)
  //   optimization: if in-order, don't copy
  // recurse: on each disjoint subset
  // leaf: transfer segment to BackStack
  // post: remove any slack from shrinking

  using namespace ot;
  constexpr int dim = DIM;
  constexpr int nbuckets = nchild(dim);  // just children, no parent
  using Buckets = std::array<size_t, nbuckets + 1>;
  Buckets sfc_buckets = {};
  const size_t range_size = range[1] - range[0];

  // future: Can determine if already sorted using min and max for each child.
  // In permuted order, if always max[prev] < min[next], then sorted.

  // Use output as a buffer at nonleafs or as the destination at leafs.
  const size_t output_pre_size = output.size();
  const BackStackIterator begin_it = output.end();
  output.insert_back(&input[range[0]], &input[range[1]]);
  input.pop_front(range_size);  // input is transfered
  const BackStackIterator end_it = output.end();
  const Triangle *begin = &output[begin_it],  *end = &output[end_it];

  // Leaf of base, switch to sub tier
  Bounds octant_bounds =
      frame_bounds_local_to_global(tree_node_to_local_bounds(octant), frame);
  if (not do_refine_base(octant, octant_bounds, &input[range[0]], 0, range[1] - range[0]))
  {
    base_tier.push_back(octant);
    return triangle_tree_sub(begin, end, sub_tier, octant, sfc);
  }

  // Only intercepted elements need to assign subsets to children.
  bool retain = true;
  if (range_size > 0)
  {
    Bounds child_bounds[8];
    for (int i = 0; i < nchild(dim); ++i)
      child_bounds[i] = frame_bounds_local_to_global(tree_node_to_local_bounds(
            octant.getChildMorton(i)), frame);

    // Count and permute offsets (inclusive prefix sums)
    std::array<size_t, nbuckets> offsets = {};
    for (const Triangle *tri = begin; tri != end; ++tri)
      for (int i = 0; i < nchild(dim); ++i)
        offsets[i] += bool(intercepts(*tri, child_bounds[i]));
    for (sfc::SubIndex s(1); s < nchild(dim); ++s)
      offsets[sfc.child_num(s)] += offsets[sfc.child_num(s.minus(1))];
    const size_t inflated_size = offsets[sfc.child_num(sfc::SubIndex(nchild(dim) - 1))];

    // To skip void children, perform pointwise in/out test at parent centroid.
    const std::array<uint32_t, dim> centroid_coord = octant.range().centroid().coords();
    retain = is_interior(begin, end, centroid_coord, frame);

    // Re-push (inflated) disjoint subsets to input.
    input.resize(input.size() + inflated_size);
    Triangle *inflated_tris = &input[input.begin()];
    for (const Triangle *tri = end; tri-- != begin; )  // backward
      for (int i = 0; i < nchild(dim); ++i)
        if (intercepts(*tri, child_bounds[i]))
          inflated_tris[--offsets[i]] = *tri;

    // Clear the part of the output used as a buffer.
    output.resize(output_pre_size);

    for (sfc::SubIndex s(0); s < nchild(dim); ++s)    // Permute the buckets.
      sfc_buckets[s] = offsets[sfc.child_num(s)];
    sfc_buckets[nbuckets] = inflated_size;
  }

  // Recurse
  const FrontStackIterator parent_begin = input.begin();
  for (sfc::SubIndex s(0); s < nchild(dim); ++s)
  {
    const size_t begin = sfc_buckets[s], end = sfc_buckets[s.plus(1)];
    const bool intercepted = begin < end;
    TreeNode<uint32_t, dim> child_octant = octant.getChildMorton(sfc.child_num(s));
    child_octant.setIsOnTreeBdry(intercepted);

    if (intercepted or retain)
      triangle_tree(input, {parent_begin + begin, parent_begin + end}, output, base_tier, sub_tier,
          frame, child_octant, sfc.subcurve(s));
  }
}

std::tuple<
  std::vector<Triangle>,
  std::vector<ot::TreeNode<uint32_t, DIM>>,
  std::vector<ot::TreeNode<uint32_t, DIM>> >
refine(std::vector<Triangle> input, Frame frame)
{
  FrontStack<Triangle> input_stack(std::move(input));
  BackStack<Triangle> output_stack;

  std::vector<ot::TreeNode<uint32_t, DIM>> base_tier;
  std::vector<ot::TreeNode<uint32_t, DIM>> sub_tier;

  triangle_tree(
      input_stack, {input_stack.begin(), input_stack.end()}, output_stack,
      base_tier, sub_tier, frame, {}, ot::SFC_State<DIM>::root());

  std::vector<Triangle> output = std::move(output_stack).to_vector();

  return {output, base_tier, sub_tier};
}




// ============================================================================
// main()
// ============================================================================

#include <iostream>
#include <fstream>

int main(int argc, char * argv[])
{
  _InitializeHcurve(DIM);

  // Input/output binary STL file format
  std::ifstream input_file;
  if (argc > 1)
  {
    std::cerr << "Opening as stl file: " << argv[1] << "\n";
    input_file.open(argv[1]);
  }
  stl_trimesh trimesh = stl_trimesh_create_from_file(argc > 1 ? input_file : std::cin);
  input_file.close();

  Bounds bounds = stl_trimesh_bounds(trimesh);
  std::cerr << "Triangle bounds == " << bounds << "\n";
  bounds = bounds_grow_factor(bounds, 129.0 / 128.0);
  std::cerr << "Expanded bounds == " << bounds << "\n";
  bounds = bounds_cube(bounds);
  std::cerr << "Cubed bounds == " << bounds << "\n";

  const Frame frame = bounds_frame(bounds);

  Octree octree = octree_create_base();

  std::vector<Triangle> output,  input = stl_extract_triangles(trimesh);
  std::cerr << "after extracting, input.size()==" << input.size() << "\n";
  std::vector<ot::TreeNode<uint32_t, DIM>> base_tier, sub_tier;
  std::tie(output, base_tier, sub_tier) = refine(input, frame);

  Octree octree_refined = {base_tier};

  const bool wireframe_mode = false;
  auto octree_trimesh = octree_to_trimesh(octree_refined, frame, wireframe_mode);
  Bounds octree_bounds = stl_trimesh_bounds(octree_trimesh);
  std::cerr << "Octree bounds == " << octree_bounds << "\n";
  std::cerr << "Octree octants == " << octree_refined.tree_nodes.size() << "\n";
  std::cerr << "Octree triangles == " << octree_trimesh.frontmatter.n_triangles << "\n";
  std::cerr << "Octree final data size == " << output.size() << "\n";
  stl_trimesh_to_file(std::cout, octree_trimesh);
  stl_trimesh_destroy(octree_trimesh);

  std::cerr << "\n";
  std::cerr << "_____________________________________________________\n";
  std::cerr << "Counters:\n";
  const auto logging = spct::tagged<LogTriIntercept>();
  const int total = logging.ref(0, total);
  for (int i = 0; i < logging.number_of_references(); ++i)
  {
    fprintf(stderr, "%20s: %10d (%.0f%%)\n",
        logging.name(i).c_str(), logging.ref(i, int{}), 100.0 * logging.ref(i, int{}) / total);
  }

  /// stl_trimesh_to_file(std::cout, trimesh);
  stl_trimesh_destroy(trimesh);

  _DestroyHcurve();

  return 0;
}
