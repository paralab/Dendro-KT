
#include <parUtils.h>

#include <set>
#include <utility>
#include <algorithm>
#include <random>
#include <sstream>

struct Graph
{
  public:
    using Edge = std::pair<int, int>;
    using EdgeList = std::vector<Edge>;
    using Range = std::pair<int, int>;

    static Graph Random(int n_vertices, int n_edges);

    Graph(int n_vertices, EdgeList edges);

    int n_vertices() const;
    int n_edges() const;
    int n_dest(int vertex) const;
    int n_src(int vertex) const;
    int dest(int vertex, int idx) const;
    int src(int vertex, int idx) const;

  private:
    int m_vertices;
    EdgeList m_edges;
    EdgeList m_reversed_edges;

    std::vector<Range> m_adjacency;
    std::vector<Range> m_reversed_adjacency;

    static EdgeList reversed(EdgeList edges);
    static std::vector<Range> build_adjacency(int n_vertices, const EdgeList &edges);
};

void assert_graph_consistent(const Graph &graph);


//
// main()
//
int main(int argc, char * argv[])
{
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  int comm_size,  comm_rank;
  MPI_Comm_size(comm, &comm_size);
  MPI_Comm_rank(comm, &comm_rank);

  const double k_avg = 2;
  const int edges = k_avg * comm_size;
  const auto graph = Graph::Random(comm_size, edges);

  assert(graph.n_vertices() == comm_size);
  assert(graph.n_edges() == edges);
  assert_graph_consistent(graph);
  if (comm_rank == 0)
  {
    std::stringstream ss_fwd,  ss_bak;
    for (int v = 0; v < graph.n_vertices(); ++v)
    {
      ss_fwd << v << "->[";
      for (int ve = 0; ve < graph.n_dest(v); ++ve)
        ss_fwd << graph.dest(v, ve) << " ";
      ss_fwd << "]  ";
    }
    for (int v = 0; v < graph.n_vertices(); ++v)
    {
      ss_bak << "[";
      for (int ve = 0; ve < graph.n_src(v); ++ve)
        ss_bak << graph.src(v, ve) << " ";
      ss_bak << "]->" << v << "  ";
    }
    printf("Graph(N=%d, E=%d):  %s\nReversed:  %s\n",
        graph.n_vertices(),
        graph.n_edges(),
        ss_fwd.str().c_str(),
        ss_bak.str().c_str());
  }

  std::vector<int> dest,  src;
  std::vector<int> send_scalar,  recv_scalar;

  const int count = 2;
  const auto map_item = [](int rank, int item) { return (rank + 1) * (item + 1); };

  for (int dest_idx = 0; dest_idx < graph.n_dest(comm_rank); ++dest_idx)
  {
    dest.push_back(graph.dest(comm_rank, dest_idx));
    for (int item = 0; item < count; ++item)
      send_scalar.push_back(map_item(graph.dest(comm_rank, dest_idx), item));
  }

  par::Mpi_NBX(dest.data(), dest.size(), send_scalar.data(), count, src, recv_scalar, comm);


  assert(src.size() == graph.n_src(comm_rank));
  for (int src_idx = 0; src_idx < graph.n_src(comm_rank); ++src_idx)
    assert(src[src_idx] == graph.src(comm_rank, src_idx));
  assert(recv_scalar.size() == graph.n_src(comm_rank) * count);
  for (int src_idx = 0; src_idx < graph.n_src(comm_rank); ++src_idx)
  {
    for (int item = 0; item < count; ++item)
      assert(recv_scalar[src_idx * count + item] == map_item(comm_rank, item));
  }

  MPI_Barrier(comm);
  if (comm_rank == 0)
    printf("success\n");  // didn't crash from any assert failure

  MPI_Finalize();
  return 0;
}


// Graph::Random()
Graph Graph::Random(int n_vertices, int n_edges)
{
  std::mt19937_64 gen;
  std::uniform_int_distribution<int> d(0, n_vertices - 1);

  std::set<Edge> e;
  const int n_squared = n_vertices * n_vertices;
  while (e.size() < n_squared and e.size() < n_edges)
    e.insert(Edge{d(gen), d(gen)});

  return Graph(n_vertices, EdgeList(e.begin(), e.end()));
}

// Graph::Graph()
Graph::Graph(int n_vertices, EdgeList edges)
  : m_vertices(n_vertices),
    m_edges(edges),
    m_reversed_edges(reversed(std::move(edges)))
{
  assert(std::is_sorted(m_edges.begin(), m_edges.end()));
  assert(std::is_sorted(m_reversed_edges.begin(), m_reversed_edges.end()));
  assert(m_edges.size() == m_reversed_edges.size());
  m_adjacency = build_adjacency(n_vertices, m_edges);
  m_reversed_adjacency = build_adjacency(n_vertices, m_reversed_edges);
}

// Graph::n_vertices()
int Graph::n_vertices() const { return m_vertices; }

// Graph::n_edges()
int Graph::n_edges() const { return m_edges.size(); }

// Graph::n_dest()
int Graph::n_dest(int vertex) const {
  assert(0 <= vertex and vertex < m_vertices);
  const Range r = m_adjacency[vertex];
  return r.second - r.first;
}

// Graph::n_src()
int Graph::n_src(int vertex) const {
  assert(0 <= vertex and vertex < m_vertices);
  const Range r = m_reversed_adjacency[vertex];
  return r.second - r.first;
}

// Graph::dest()
int Graph::dest(int vertex, int idx) const {
  assert(0 <= vertex and vertex < m_vertices);
  assert(0 <= idx and idx < n_dest(vertex));
  const Edge e = m_edges[m_adjacency[vertex].first + idx];
  assert(e.first == vertex);
  return e.second;
}

// Graph::src()
int Graph::src(int vertex, int idx) const {
  assert(0 <= vertex and vertex < m_vertices);
  assert(0 <= idx and idx < n_src(vertex));
  const Edge e = m_reversed_edges[m_reversed_adjacency[vertex].first + idx];
  assert(e.first == vertex);
  return e.second;
}

// Graph::reversed()
Graph::EdgeList Graph::reversed(EdgeList edges)
{
  for (Edge &e : edges)
  {
    const int src = e.first,  dest = e.second;
    e.first = dest;
    e.second = src;
  }

  std::sort(edges.begin(), edges.end());
  return edges;
}

// Graph::build_adjacency()
std::vector<Graph::Range> Graph::build_adjacency(int n_vertices, const EdgeList &edges)
{
  std::vector<Range> range(n_vertices, {0, 0});

  int begin = 0, end = 0;
  for (int v = 0; v < n_vertices; ++v)
  {
    for (; begin < edges.size() and edges[begin].first < v; ++begin);
    end = begin;
    for (; end < edges.size() and edges[end].first == v; ++end);
    range[v] = {begin, end};
    begin = end;
  }

  return range;
}



// assert_graph_consistent()
void assert_graph_consistent(const Graph &graph)
{
  {
    int total_dest = 0,  total_src = 0;
    for (int v = 0; v < graph.n_vertices(); ++v)
    {
      total_dest += graph.n_dest(v);
      total_src += graph.n_src(v);
    }
    assert(total_dest == graph.n_edges());
    assert(total_src == graph.n_edges());
  }
  {
    for (int v = 0; v < graph.n_vertices(); ++v)
    {
      const int outdeg = graph.n_dest(v);
      for (int idx = 0; idx < outdeg; ++idx)
      {
        const int w = graph.dest(v, idx);
        assert(w < graph.n_vertices());
        const int w_src = graph.n_src(w);
        assert(w_src > 0);
        bool found = false;
        for (int w_idx = 0; w_idx < w_src; ++w_idx)
          if (graph.src(w, w_idx) == v)
            found = true;
        assert(found);
      }
    }
  }

}
