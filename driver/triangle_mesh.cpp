
// ----------------------------------------------------------------------------
// Point
// ----------------------------------------------------------------------------

#include <stdint.h>

struct Vec3f
{
  float data[3] = {};

  constexpr float operator[](int d) const
  {
    return data[d];
  }
};

struct Point3f
{
  float data[3] = {};

  constexpr float operator[](int d) const
  {
    return data[d];
  }
};

struct Point3u
{
  uint32_t data[3] = {};

  constexpr uint32_t operator[](int d) const
  {
    return data[d];
  }
};


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
    bounds.ranges[d] = range_set_length(bounds.ranges[d], max_length);
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

stl_tri sti_trimesh_get_tri(const stl_trimesh &trimesh, int i)
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



// ----------------------------------------------------------------------------
// octree to stl
// ----------------------------------------------------------------------------

#include <array>

stl_trimesh octree_to_trimesh(const Octree &octree, const Frame frame)
{
  const auto n_octants = octree.tree_nodes.size();
  const auto n_faces = n_octants * 6;
  const auto n_triangles = n_faces * 2;

  stl_trimesh trimesh = stl_trimesh_create(stl_frontmatter_create(n_triangles));

  int triangle_idx = 0;
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

    // Emit triangles (with right-hand-rule) over octant surface.

    // Touching vertex 0
    stl_trimesh_set_tri(trimesh, triangle_idx++, make_tri({0, 1, 5}, 2));
    stl_trimesh_set_tri(trimesh, triangle_idx++, make_tri({0, 5, 4}, 2));
    stl_trimesh_set_tri(trimesh, triangle_idx++, make_tri({0, 4, 6}, 0));
    stl_trimesh_set_tri(trimesh, triangle_idx++, make_tri({0, 6, 2}, 0));
    stl_trimesh_set_tri(trimesh, triangle_idx++, make_tri({0, 2, 3}, 4));
    stl_trimesh_set_tri(trimesh, triangle_idx++, make_tri({0, 3, 1}, 4));

    // Touching vertex 7
    stl_trimesh_set_tri(trimesh, triangle_idx++, make_tri({5, 1, 7}, 1));
    stl_trimesh_set_tri(trimesh, triangle_idx++, make_tri({4, 5, 7}, 5));
    stl_trimesh_set_tri(trimesh, triangle_idx++, make_tri({6, 4, 7}, 5));
    stl_trimesh_set_tri(trimesh, triangle_idx++, make_tri({2, 6, 7}, 3));
    stl_trimesh_set_tri(trimesh, triangle_idx++, make_tri({3, 2, 7}, 3));
    stl_trimesh_set_tri(trimesh, triangle_idx++, make_tri({1, 3, 7}, 1));
  }

  return trimesh;
}



// ============================================================================
// main()
// ============================================================================

#include <iostream>

int main(int argc, char * argv[])
{
  // Input/output binary STL file format
  stl_trimesh trimesh = stl_trimesh_create_from_file(std::cin);

  Bounds bounds = stl_trimesh_bounds(trimesh);
  std::cerr << "Triangle bounds == " << bounds << "\n";
  bounds = bounds_grow_factor(bounds, 129.0 / 128.0);
  std::cerr << "Expanded bounds == " << bounds << "\n";
  bounds = bounds_cube(bounds);
  std::cerr << "Cubed bounds == " << bounds << "\n";

  const Frame frame = bounds_frame(bounds);

  Octree octree = octree_create_base();

  auto octree_trimesh = octree_to_trimesh(octree, frame);
  Bounds octree_bounds = stl_trimesh_bounds(octree_trimesh);
  std::cerr << "number of triangles == " << octree_trimesh.frontmatter.n_triangles << "\n";
  std::cerr << "Octree bounds == " << octree_bounds << "\n";
  stl_trimesh_to_file(std::cout, octree_trimesh);
  stl_trimesh_destroy(octree_trimesh);

  /// stl_trimesh_to_file(std::cout, trimesh);
  stl_trimesh_destroy(trimesh);

  return 0;
}
