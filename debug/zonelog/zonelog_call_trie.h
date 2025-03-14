#ifndef ZONELOG_CALL_TRIE_H
#define ZONELOG_CALL_TRIE_H

#include "zonelog.h"

#include <map>
#include <string_view>
#include <stdexcept>
#include <ranges>

namespace zonelog
{
  class SumCalls;

  // -----------------------------------------------------------------------
  // Radix tree / trie on comparable data
  // -----------------------------------------------------------------------
  template <typename Key>
  class Trie;

  // A node is uniquely defined as a sequence of keys.
  // But we assign a 64-bit hash of the sequence as its name.
  // Detect collisions by inspecting whether parents match.

  // map arbitrary data to nodes in the tree.
  //   a different mapping data structure can be used for this part.

  // Key must be a standard layout type without padding.
  // And the alignment must be equal to that of size_t.
  template <typename Key>
  size_t hash_bytes(Key key)
  {
    return std::hash<std::string_view>{}(
        std::string_view(
          reinterpret_cast<char *>(&key),
          sizeof(key)));
  };


  // -------------------------------------------------------------------------
  //
  // Tree: Tree in which children are unsequenced.
  // Treq: Tree in which children are SEQuenced.
  // Trap: Tree in which children are labeled by unique dictionary (MAP) keys.
  //
  // -------------------------------------------------------------------------
  //
  // Treq[T] = (root: T, children: Seq[Treq[T]])
  //
  // pre_nodes: Treq[T] -> Seq[T] :=
  //   (root: T, children: Seq[Treq[T]]) -> seq(root) ++ (join map depth_pre_order children)
  //
  // post_nodes: Treq[T] -> Seq[T] :=
  //   (root: T, children: Seq[Treq[T]]) -> (join map depth_pre_order children) ++ seq(root)
  //
  // leafs: Treq[T] -> Seq[T] :=
  //   (root: T, children: Empty) -> seq(root)
  //   (root: T, children: Seq[Treq[T]]) -> join map leafs children
  //
  // levels: Treq[T] -> Seq[Seq[T]]  # zip is transpose-like, refolding
  //   (root: T, children: Empty) -> seq(seq(root))
  //   (root: T, children: Seq[Treq[T]]) -> seq(seq(root)) ++ ziplong (map levels children)
  //
  // where
  //   zip2 Seq[A] Seq[B] -> Seq[(A,B)] :=
  //     (head_a, tail_a) (head_b, tail_b) -> ((head_a, head_b), zip tail_a tail_b)
  //     Empty seq_b -> Empty
  //     seq_a Empty -> Empty
  //
  //   ziplong2 Seq[A] Seq[B] -> Seq[(A,B) or (A) or (B)] :=
  //     (head_a, tail_a) (head_b, tail_b) -> ((head_a, head_b), zip tail_a tail_b)
  //     Empty (head_b tail_b) -> (seq(head_b), zip tail_b)
  //     (head_a tail_a) Empty -> (seq(head_a), zip tail_a)
  //
  //   ziplong Seq[Seq[T]] -> Seq[Seq[T]] :=  # incidentally, 0-aligns rows
  //     seqs ->
  //       let nonemptys = filter (not eq Empty) seqs
  //       in (map head nonemptys, map (ziplong tail) nonemptys)
  //
  // -------------------------------------------------------------------------
  //
  // Tree[T] = (root: T, children: Set[Tree[T]])
  // - The same as Treq[T] but with Seq replaced by Set.
  // - leafs produces the set of leafs.
  // - pre_nodes and post_nodes degenerate to just nodes: the set of nodes.
  //
  // Trap[K->T] = (root: T, children: Map[K -> Trap[K->T]])
  // - Similar to Treq[T], degenerates to Tree[T] by (values: Map[K->X] -> Set[X])
  //
  // The compact root-first structure emphasizes nesting and preemptively disallows cycles.
  //
  // A tree could also be embedded into a map sending children to parents.
  // (Such maps have enough generality to embed forests or directed graphs.)
  //   Tree[T] ⊂ (nodes: Multiset[T], leafs: Set[nodes], parent: Map[nodes -> Maybe nodes])
  //

  // A trie is isomorphic to an associative rooted tree (Trap[T] above):
  //   Trie nodes are the chains of tree nodes starting* from the root,
  //   and trie parents are the prefixes* of tree paths.
  //
  // chains: Trap[K->T] -> Trap[K->(T, Seq[K])] :=
  //   (root: T, children: Map())
  //     -> (root: (root, Seq()), children: Map())
  //   (root: T, children: Map[K -> Trap[K->T]])
  //     -> (root: (root, Seq()), children: { k: tree_map (prefix_chain k) subtree | {k: subtree} in children })
  //
  // where
  //   treemap f (root, children) := ((f root), map (treemap f) children)
  //
  //   prefix_chain k (datum, chain) := (datum, seq(k)++chain)
  //
  //
  // *(Or, suffixes, ending at the root, to say parent(node)=tail(path).)
  //
  //  make_trie: (nodes, leafs: Set[nodes], tree_parent: Map[nodes -> Maybe nodes])
  //    -> (paths: Set[Seq[nodes]], maximal_paths: Set[paths], trie_parent: Map[paths -> Maybe paths]) :=
  //
  //    let path (n: nodes) := (seq(n)++(path(tree_parent(n)) or Empty))
  //    in
  //      paths := image path nodes
  //      maximal_paths := image path leafs
  //      trie_parent: paths -> Maybe paths :=
  //        (node, ancestors) -> ancestors
  //        Empty -> Nothing
  //
  // (Unfortunately, modeling each path as a sequence of _nodes_, rather than
  // of links, results in an off-by-one correspondence, where there is
  // a superoot in a trie for an empty tree. In practice, when constructing a
  // trie, it must be supposed that keys are stored on branches and values are
  // stored on non-root nodes.)


  template <typename Key>
  class Trie
  {
    public:
      static constexpr size_t root() { return hash_bytes(Key()); }

      static constexpr bool is_root(size_t node)
      {
        return node == root();
      }

      struct Branch { size_t parent; Key key; };
      constexpr size_t branch(Branch branch)
      {
        static_assert(alignof(decltype(Branch::parent)) == alignof(decltype(Branch::key)));
        const size_t node_hash = hash_bytes(branch);  // beware of padding!

        if (node_hash == root())
        {
          throw::std::range_error("Hash collision in Trie with root!");
        }

        if (not map.contains(node_hash))
        {
          map[node_hash] = NodeProperty{ branch.parent, branch.key };
        }
        else if (map.at(node_hash).parent != branch.parent)
        {
          throw std::range_error("Hash collision in Trie");
        }

        return node_hash;
      }

      constexpr size_t parent(size_t node) const
      {
        if (is_root(node))
          throw std::invalid_argument("Tried to obtain parent of root().");

        return map.at(node).parent;
      }

      constexpr Key key(size_t node) const
      {
        return map.at(node).key;
      }

      constexpr auto view() const
      {
        return map |
            std::views::transform(
                [](const auto &n_v){
                    return std::make_pair(n_v.first, n_v.second.key);
                });
      }
    private:
      struct NodeProperty { size_t parent; Key key; };  // add children for traversal
      std::map<size_t, NodeProperty> map;
  };


  // -----------------------------------------------------------------------


  class SumCalls
  {
    public:
      inline void consume_event(Event event);
      void print_results() const;
    private:
      struct CodePathProperty {
        long int count = 0;
        clock::duration duration = {};
        clock::time_point last_entry = {};

        void open(clock::time_point when) { ++count; last_entry = when; }
        void close(clock::time_point when) { duration += (when - last_entry); }
        // Never open twice before closing. Guaranteed by call_trie.
      };

      struct CallProperty {
        ZonePtrWData zone_w_data = {};
        long int count = 0;
        clock::duration duration = {};
        clock::duration self_duration = {};

        CallProperty & operator+=(const CallProperty &y)
        {
          zone_w_data    = y.zone_w_data;
          count         += y.count;
          duration      += y.duration;
          self_duration += y.self_duration;
          return *this;
        }
      };

      std::map<size_t, CodePathProperty> code_path_properties;

      Trie<ZonePtrWData> call_trie;
      size_t open_code_path = call_trie.root();
  };

  static_assert(offline::LogAggregator<SumCalls>);


  void SumCalls::consume_event(Event event)
  {
    // assume the input data comes on push
    if (is_push(event.mark.zone_action))
    {
      const uintptr_t zone_code = zonelog::zone_code(event.mark.zone_action);
      const uint64_t has_data = zonelog::has_data(event.mark.zone_action);
      const EventData data {event.data};
      const size_t parent = open_code_path;
      const size_t code_path =
          call_trie.branch({parent, ZonePtrWData{zone_code, data, has_data}});
      code_path_properties[code_path].open(event.mark.time_stamp);
      open_code_path = code_path;
    }
    else
    {
      const uintptr_t zone_code = zonelog::zone_code(event.mark.zone_action);
      if (zone_code != call_trie.key(open_code_path).zone_code)
        throw std::logic_error("Attempting to close a zone that is blocked or not open.");
      code_path_properties[open_code_path].close(event.mark.time_stamp);
      open_code_path = call_trie.parent(open_code_path);
    }
  }

}

#endif//ZONELOG_CALL_TRIE_H
