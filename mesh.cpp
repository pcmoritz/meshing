#include <vector>
#include <iostream>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>
#include <CGAL/Mesh_polyhedron_3.h>
#include <CGAL/Polyhedral_mesh_domain_with_features_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Triangulation_cell_base_with_info_3.h>
#include <CGAL/make_mesh_3.h>

#include <dolfin/common/MPI.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEditor.h>
#include <dolfin/mesh/MeshPartitioning.h>

#define HAS_CGAL

#include <dolfin/generation/CGALMeshBuilder.h>
#include "cgal_triangulate_polyhedron.h"
#include "cgal_csg3d.h"

#include <dolfin/plot/plot.h>

using namespace dolfin;

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;

typedef CGAL::Polyhedral_mesh_domain_with_features_3<K> Mesh_domain;

typedef CGAL::Robust_weighted_circumcenter_filtered_traits_3<K> Geom_traits;

// CGAL 3D triangulation vertex typedefs
typedef CGAL::Triangulation_vertex_base_3<Geom_traits> Tvb3test_base;
typedef CGAL::Triangulation_vertex_base_with_info_3<int, Geom_traits, Tvb3test_base> Tvb3test;
typedef CGAL::Mesh_vertex_base_3<Geom_traits, Mesh_domain, Tvb3test> Vertex_base;

// CGAL 3D triangulation cell typedefs
typedef CGAL::Triangulation_cell_base_3<Geom_traits> Tcb3test_base;
typedef CGAL::Triangulation_cell_base_with_info_3<int, Geom_traits, Tcb3test_base> Tcb3test;
typedef CGAL::Mesh_cell_base_3<Geom_traits, Mesh_domain, Tcb3test> Cell_base;

// CGAL 3D triangulation typedefs
typedef CGAL::Triangulation_data_structure_3<Vertex_base, Cell_base> Tds_mesh;
typedef CGAL::Regular_triangulation_3<Geom_traits, Tds_mesh> Tr;

// CGAL 3D mesh typedef
typedef CGAL::Mesh_complex_3_in_triangulation_3<
  Tr, Mesh_domain::Corner_index, Mesh_domain::Curve_segment_index> C3t3;

// Mesh criteria
typedef CGAL::Mesh_criteria_3<Tr> Mesh_criteria;

// Typedefs for building CGAL polyhedron from list of facets
typedef CGAL::Mesh_polyhedron_3<K>::Type Polyhedron;
typedef Polyhedron::Facet_iterator Facet_iterator;
typedef Polyhedron::Halfedge_around_facet_circulator Halfedge_facet_circulator;
typedef Polyhedron::HalfedgeDS HalfedgeDS;

template<typename T>
void cgal_generate(Mesh& mesh, T& p, double cell_size,
		   bool detect_sharp_features)
{
  // Check if any facets are not triangular and triangulate if necessary.
  // The CGAL mesh generation only supports polyhedra with triangular surface
  // facets.

  typename Polyhedron::Facet_iterator facet;
  for (facet = p.facets_begin(); facet != p.facets_end(); ++facet)
  {
    // Check if there is a non-triangular facet
    if (!facet->is_triangle())
    {
      CGAL::triangulate_polyhedron(p);
      break;
    }
  }

  // Create domain from polyhedron
  Mesh_domain domain(p);

  // Get sharp features
  if (detect_sharp_features)
    domain.detect_features();

  const Mesh_criteria criteria(CGAL::parameters::facet_angle = 25,
                               CGAL::parameters::facet_size = cell_size,
                               CGAL::parameters::cell_radius_edge_ratio = 3.0,
                               CGAL::parameters::edge_size = cell_size);

  // Generate CGAL mesh
  C3t3 c3t3 = CGAL::make_mesh_3<C3t3>(domain, criteria);

  // Build DOLFIN mesh from CGAL mesh/triangulation
  CGALMeshBuilder::build_from_mesh(mesh, c3t3);
}




int main() {
  std::string off_file = "../cube.off";
  Polyhedron p;
  std::cout << "reading file " << off_file << std::endl;
  std::ifstream p_file(off_file.c_str());
  p_file >> p;
  std::cout << "done reading file." << std::endl;
  double cell_size = 0.5;
  bool detect_sharp_features = true;
  Mesh mesh;

  cgal_generate(mesh, p, cell_size, detect_sharp_features);
  plot(mesh, "mesh of a cube");
  interactive(true);
  std::getchar();
}
