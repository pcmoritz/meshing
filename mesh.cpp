#include <vector>
#include <iostream>
#include <cmath>

#include <CGAL/Nef_polyhedron_3.h>
#include <CGAL/Aff_transformation_3.h>
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

#include <dolfin/plot/plot.h>

#define HAS_CGAL

#include <dolfin/generation/CGALMeshBuilder.h>

#include "cgal_triangulate_polyhedron.h"
#include "cgal_copy_polygon_to.h"
#include "cgal_csg3d.h"

using namespace dolfin;

void build_mesh(const csg::C3t3& c3t3, Mesh& mesh)
{
  typedef csg::C3t3 C3T3;
  typedef C3T3::Triangulation Triangulation;
  typedef Triangulation::Vertex_handle Vertex_handle;

  // CGAL triangulation
  const Triangulation& triangulation = c3t3.triangulation();

  // Clear mesh
  mesh.clear();

  // Count cells in complex
  std::size_t num_cells = 0;
  for(csg::C3t3::Cells_in_complex_iterator cit = c3t3.cells_in_complex_begin();
      cit != c3t3.cells_in_complex_end();
      ++cit)
  {
    num_cells++;
  }

  // Create and initialize mesh editor
  dolfin::MeshEditor mesh_editor;
  mesh_editor.open(mesh, 3, 3);
  mesh_editor.init_vertices(triangulation.number_of_vertices());
  mesh_editor.init_cells(num_cells);

  // Add vertices to mesh
  std::size_t vertex_index = 0;
  std::map<Vertex_handle, std::size_t> vertex_id_map;

  for (Triangulation::Finite_vertices_iterator
         cgal_vertex = triangulation.finite_vertices_begin();
       cgal_vertex != triangulation.finite_vertices_end(); ++cgal_vertex)
  {
    vertex_id_map[cgal_vertex] = vertex_index;

      // Get vertex coordinates and add vertex to the mesh
    Point p(cgal_vertex->point()[0], cgal_vertex->point()[1], cgal_vertex->point()[2]);
    mesh_editor.add_vertex(vertex_index, p);

    ++vertex_index;
  }

  // Add cells to mesh
  std::size_t cell_index = 0;
  for(csg::C3t3::Cells_in_complex_iterator cit = c3t3.cells_in_complex_begin();
      cit != c3t3.cells_in_complex_end();
      ++cit)
  {
    mesh_editor.add_cell(cell_index,
                         vertex_id_map[cit->vertex(0)],
                         vertex_id_map[cit->vertex(1)],
                         vertex_id_map[cit->vertex(2)],
                         vertex_id_map[cit->vertex(3)]);

    ++cell_index;
  }

  // Close mesh editor
  mesh_editor.close();
}

void generate(Mesh& mesh, const csg::Polyhedron_3& p, double cell_size) {
  dolfin_assert(p.is_pure_triangle());
  csg::Mesh_domain domain(p);
  domain.detect_features();
  csg::Mesh_criteria criteria(CGAL::parameters::facet_angle = 25,
			      CGAL::parameters::facet_size = cell_size,
			      CGAL::parameters::cell_radius_edge_ratio = 3.0,
			      CGAL::parameters::edge_size = cell_size);
  csg::C3t3 c3t3 = CGAL::make_mesh_3<csg::C3t3>(domain, criteria,
                                                CGAL::parameters::no_perturb(),
                                                CGAL::parameters::no_exude());
  build_mesh(c3t3, mesh);
}



// Apply a transformation of the form QP + t to the polyhedron P where
// Q is the rotation given by a quaternion (x, y, z, w) and t is a
// translation vector.
csg::Exact_Polyhedron_3
transformed(csg::Exact_Polyhedron_3& P,
	    double x, double y, double z,
	    double w, double tx, double ty, double tz) {
  double s = std::sin(w);
  double c = std::cos(w);
  // Rotation
  CGAL::Aff_transformation_3<csg::Exact_Kernel>
    R(c + x*x*(1-c), x*y*(1-c) - z*s, x*z*(1-c) + y*s,
      y*x*(1-c) + z*s, c + y*y*(1-c), y*z*(1-c) - x*s,
      z*x*(1-c) - y*x, z*y*(1-c) + x*s, c + z*z*(1-c));
  // Translation
  CGAL::Aff_transformation_3<csg::Exact_Kernel>
    T(1, 0, 0, tx, 0, 1, 0, ty, 0, 0, 1, tz);
  std::transform(P.points_begin(), P.points_end(), P.points_begin(), T*R);
  return P;
}


int main() {
  std::string off_file = "../cube.off";
  csg::Exact_Polyhedron_3 cube; // unit cube
  std::cout << "reading file " << off_file << std::endl;
  std::ifstream file(off_file.c_str());
  file >> cube;
  std::cout << "done reading file." << std::endl;
  double cell_size = 0.5;
  bool detect_sharp_features = true;
  Mesh mesh;
  
  csg::Exact_Polyhedron_3 outer(cube);
  
  // scale and translate the outer box
  CGAL::Aff_transformation_3<csg::Exact_Kernel> S(4, 0, 0, -1, 0, 3, 0, -1, 0, 0, 2.5, -1);
  std::transform(outer.points_begin(), outer.points_end(), 
  		 outer.points_begin(), S);

  csg::Nef_polyhedron_3 Omega(outer);
  csg::Exact_Polyhedron_3 first_inner(cube);
  csg::Exact_Polyhedron_3 second_inner(cube);
  second_inner = transformed(second_inner, 0, 0, 1, 0.1, 1.5, 0, 0);
  
  Omega -= first_inner;
  Omega -= second_inner;

  csg::Exact_Polyhedron_3 p;
  Omega.convert_to_polyhedron(p);
  csg::Polyhedron_3 q;
  copy_to(p, q);
  

  generate(mesh, q, cell_size);
  plot(mesh, "mesh of a cube");
  interactive(true);
  std::getchar();
}
