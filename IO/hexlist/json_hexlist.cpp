//
// Created by masado on 4/26/22.
//

#include "json_hexlist.h"

namespace io
{
  // hex2oct()
  template <typename T, unsigned dim>
  static ot::TreeNode<T, dim> hex2oct(const JSON_Hexlist::Hex &hex, int unit_level);

  // oct2hex()
  template <typename T, unsigned dim>
  static JSON_Hexlist::Hex oct2hex(const ot::TreeNode<T, dim> &oct, int unit_level);

  // octlist_size()
  size_t JSON_Hexlist::octlist_size() const
  {
    return data.size();
  }

  // to_octlist()
  template <typename T, unsigned dim>
  void JSON_Hexlist::to_octlist(ot::TreeNode<T, dim> *octlist, int unit_level) const
  {
    for (const Hex &h: data)
      *(octlist++) = hex2oct<T, dim>(h, unit_level);
  }

  // from_octlist()
  template <typename T, unsigned dim>
  JSON_Hexlist JSON_Hexlist::from_octlist(const ot::TreeNode<T, dim> *octlist, size_t size, int unit_level)
  {
    JSON_Hexlist json_hexlist;
    json_hexlist.data.reserve(size);
    for (size_t i = 0; i < size; ++i)
      json_hexlist.data.push_back(oct2hex(octlist[i], unit_level));
    return json_hexlist;
  }

  // to_octlist()
  template <typename T, unsigned dim>
  void JSON_Hexlist::to_octlist(std::vector<ot::TreeNode<T, dim>> &octlist, int unit_level) const
  {
    octlist.resize(this->octlist_size());
    this->template to_octlist(octlist.data(), unit_level);
  }

  // from_octlist()
  template <typename T, unsigned dim>
  JSON_Hexlist JSON_Hexlist::from_octlist(const std::vector<ot::TreeNode<T, dim>> &octlist, int unit_level)
  {
    return from_octlist(octlist.data(), octlist.size(), unit_level);
  }
}


namespace io
{
  // hex2oct()
  template <typename T, unsigned dim>
  static ot::TreeNode<T, dim> hex2oct(const JSON_Hexlist::Hex &hex, int unit_level)
  {
    enum { Dkt_X, Dkt_Y, Dkt_Z };
    enum { BB_X, BB_Y, BB_Z};

    const T unit = (1u << (m_uiMaxDepth - unit_level));
    const T side = hex.size * unit;
    const int level = m_uiMaxDepth - binOp::fastLog2(side);

    ot::TreeNode<T, dim> tree_node = {};
    tree_node.setLevel(level);
    if (dim >= 1)  tree_node.setX(Dkt_X, hex.coords[BB_X] * unit);
    if (dim >= 2)  tree_node.setX(Dkt_Y, hex.coords[BB_Y] * unit);
    if (dim >= 3)  tree_node.setX(Dkt_Z, hex.coords[BB_Z] * unit);
    return tree_node;
  }

  // oct2hex()
  template <typename T, unsigned dim>
  static JSON_Hexlist::Hex oct2hex(const ot::TreeNode<T, dim> &oct, int unit_level)
  {
    enum { Dkt_X, Dkt_Y, Dkt_Z };
    enum { BB_X, BB_Y, BB_Z};

    const T unit = (1u << (m_uiMaxDepth - unit_level));
    JSON_Hexlist::Hex hex = {};
    hex.coords[BB_X] = (dim >= 1) ? oct.getX(Dkt_X) / unit : 0;
    hex.coords[BB_Y] = (dim >= 2) ? oct.getX(Dkt_Y) / unit : 0;
    hex.coords[BB_Z] = (dim >= 3) ? oct.getX(Dkt_Z) / unit : 0;
    hex.size = oct.range().side() / unit;
    return hex;
  }
}


namespace io
{
  template void JSON_Hexlist::to_octlist(ot::TreeNode<unsigned, 2> *, int) const;
  template void JSON_Hexlist::to_octlist(ot::TreeNode<unsigned, 3> *, int) const;
  template void JSON_Hexlist::to_octlist(ot::TreeNode<unsigned, 4> *, int) const;

  template JSON_Hexlist JSON_Hexlist::from_octlist(const ot::TreeNode<unsigned, 2> *, size_t, int);
  template JSON_Hexlist JSON_Hexlist::from_octlist(const ot::TreeNode<unsigned, 3> *, size_t, int);
  template JSON_Hexlist JSON_Hexlist::from_octlist(const ot::TreeNode<unsigned, 4> *, size_t, int);

  template void JSON_Hexlist::to_octlist(std::vector<ot::TreeNode<unsigned, 2>> &, int) const;
  template void JSON_Hexlist::to_octlist(std::vector<ot::TreeNode<unsigned, 3>> &, int) const;
  template void JSON_Hexlist::to_octlist(std::vector<ot::TreeNode<unsigned, 4>> &, int) const;

  template JSON_Hexlist JSON_Hexlist::from_octlist(const std::vector<ot::TreeNode<unsigned, 2u>> &, int);
  template JSON_Hexlist JSON_Hexlist::from_octlist(const std::vector<ot::TreeNode<unsigned, 3u>> &, int);
  template JSON_Hexlist JSON_Hexlist::from_octlist(const std::vector<ot::TreeNode<unsigned, 4u>> &, int);
}