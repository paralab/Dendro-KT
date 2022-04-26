//
// Created by masado on 4/26/22.
//

#ifndef DENDRO_KT_JSON_HEXLIST_H
#define DENDRO_KT_JSON_HEXLIST_H

#include "include/treeNode.h"
#include <vector>
#include "IO/vtk/include/json.hpp"

namespace io
{
  struct JSON_Hexlist
  {
    size_t octlist_size() const;  // number of octants to allocate before to_octlist()

    template <typename T, unsigned dim>
    void to_octlist(ot::TreeNode<T, dim> *octlist, int unit_level) const;  // assumes already allocated

    template <typename T, unsigned dim>
    static JSON_Hexlist from_octlist(const ot::TreeNode<T, dim> *octlist, size_t size, int unit_level);

    template <typename T, unsigned dim>
    void to_octlist(std::vector<ot::TreeNode<T, dim>> &octlist, int unit_level) const;

    template <typename T, unsigned dim>
    static JSON_Hexlist from_octlist(const std::vector<ot::TreeNode<T, dim>> &octlist, int unit_level);

    bool matching_format() const { return format_version == expected_format; }

    using HexCoord = std::array<std::uint32_t, 3>;
    using HexSize = std::uint32_t;
    struct Hex { HexCoord coords = {};  HexSize size = 1; };

    constexpr static std::array<int, 3> expected_format = {2022, 4, 26};
    std::array<int, 3>            format_version = {2022, 4, 26};
    std::array<std::string, 2>    fields = {"from", "size"};
    std::vector<Hex>              data = {};

    // Conversions like json.get<JSON_Hexlist>() are defined below.
  };

  // to_json()  of nested class Hex
  inline void to_json(nlohmann::json &j, const JSON_Hexlist::Hex &h)
  {
    j = nlohmann::json({h.coords, h.size});
  }

  // from_json()  of nested class Hex
  inline void from_json(const nlohmann::json &j, JSON_Hexlist::Hex &h)
  {
    h.coords[0] = j.at(0).at(0);
    h.coords[1] = j.at(0).at(1);
    h.coords[2] = j.at(0).at(2);
    h.size = j.at(1);
  }

  // to_json()
  inline void to_json(nlohmann::json &j, const JSON_Hexlist &hexlist)
  {
    j["format_version"] = hexlist.format_version;
    j["fields"] = hexlist.fields;
    j["data"] = hexlist.data;
  }

  inline void from_json(const nlohmann::json &j, JSON_Hexlist &hexlist)
  {
    hexlist.format_version[0] = j.at("format_version").at(0);
    hexlist.format_version[1] = j.at("format_version").at(1);
    hexlist.format_version[2] = j.at("format_version").at(2);

    hexlist.fields[0] = j.at("fields").at(0);
    hexlist.fields[1] = j.at("fields").at(1);

    hexlist.data = j.at("data").get<std::vector<JSON_Hexlist::Hex>>();
  }
}

#endif //DENDRO_KT_JSON_HEXLIST_H
