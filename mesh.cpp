// #include <vector>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <fstream>

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
#include <CGAL/Cartesian_converter.h>

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

#include <dolfin.h>
#include "model.h"

using namespace dolfin;

typedef CGAL::Simple_cartesian<double> IK;
typedef CGAL::Cartesian_converter<IK,csg::Exact_Kernel> IK_to_EK;

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

  std::cout << "Generating mesh" << std::endl;
  csg::C3t3 c3t3 = CGAL::make_mesh_3<csg::C3t3>(domain, criteria,
                                                CGAL::parameters::no_perturb(),
                                                CGAL::parameters::no_exude());
  // optimize mesh
  // std::cout << "Optimizing mesh by odt optimization" << std::endl;
  // odt_optimize_mesh_3(c3t3, domain);
  // std::cout << "Optimizing mesh by lloyd optimization" << std::endl;
  // lloyd_optimize_mesh_3(c3t3, domain);
  // This is too slow. Is it really needed?
  // std::cout << "Optimizing mesh by perturbation" << std::endl;
  // CGAL::perturb_mesh_3(c3t3, domain);
  // std::cout << "Optimizing mesh by sliver exudation" << std::endl;
  // exude_mesh_3(c3t3);

  //std::fstream out;
  //out.open("~/mesh.txt", std::ios::out);
  //out << c3t3;
  //out.close();

  build_mesh(c3t3, mesh);
}

// The transformation first scales the object by the vector (a, b, c),
// then rotates it by the quaternion (x, y, z, w) and then shifts it
// by the vector (t, u, v).
template<class K>
CGAL::Aff_transformation_3<K>
cgal_transformation(double a, double b, double c,
		    double x, double y, double z, double w,
		    double t, double u, double v)
{
  double S = std::sin(w);
  double C = std::cos(w);
  // Scaling
  CGAL::Aff_transformation_3<K> scal(a, 0, 0, 0, b, 0, 0, 0, c);
  
  // Rotation
  CGAL::Aff_transformation_3<K>
    rot(C + x*x*(1-C), x*y*(1-C) - z*S, x*z*(1-C) + y*S,
      y*x*(1-C) + z*S, C + y*y*(1-C), y*z*(1-C) - x*S,
      z*x*(1-C) - y*S, z*y*(1-C) + x*S, C + z*z*(1-C));

  // Translation
  CGAL::Aff_transformation_3<csg::Exact_Kernel>
    trans(1, 0, 0, t, 0, 1, 0, u, 0, 0, 1, v);
  
  return trans * rot * scal;
}

typedef CGAL::Aff_transformation_3<csg::Exact_Kernel> Aff_trans_3;

// Test if Point p is on the boundary of the cube which is the
// standard (-1, -1, -1) -- (1, 1, 1) cube transformed by the
// transformation t, within a precision of eps
bool is_on_boundary(const csg::Exact_Point_3& p, const Aff_trans_3& t,
		    double eps = 1e-7)
{
  csg::Exact_Point_3 q = t.inverse()(p);
  // Test if point is inside of a cube that is a bit larger
  if(!((CGAL::abs(q.x()) <= 1 + eps) &&
       (CGAL::abs(q.y()) <= 1 + eps) &&
       (CGAL::abs(q.z()) <= 1 + eps)))
    return false;
  // Test if point is not inside of a cube that is a bit smaller
  if((CGAL::abs(q.x()) <= 1 - eps) &&
     (CGAL::abs(q.y()) <= 1 - eps) &&
     (CGAL::abs(q.z()) <= 1 - eps))
    return false;
  return true;
}

bool is_inside(const Aff_trans_3& t,
	       double x, double y, double z, double eps = 1e-7)
{
  csg::Exact_Point_3 p(x, y, z);
  csg::Exact_Point_3 q = t.inverse()(p);
  // Test if point is inside of a cube that is a bit larger
  if(!((CGAL::abs(q.x()) <= 1 + eps) &&
       (CGAL::abs(q.y()) <= 1 + eps) &&
       (CGAL::abs(q.z()) <= 1 + eps)))
    return false;
  return true;
}

struct CubeDomain : public SubDomain
{
  Aff_trans_3 trans;
  std::string debug;
  CubeDomain(const Aff_trans_3& t, std::string d) 
    : trans(t), debug(d) {}
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    // std::cout << x[0] << " " << x[1] << " " << x[2];
    bool b = is_on_boundary(csg::Exact_Point_3(x[0], x[1], x[2]), trans);
    // std::cout << debug << " " << b << on_boundary << std::endl;
    return b && on_boundary;
  }
};

// Evaluate the transformation from the boundary of one cube to the
// boundary of a transformed cube.
struct CubeToCube : public Expression
{
  Aff_trans_3 first_cube;
  Aff_trans_3 second_cube;
  CubeToCube (const Aff_trans_3& a, const Aff_trans_3& b)
    : Expression(3), first_cube(a), second_cube(b) {}
  void eval(Array<double>& values, const Array<double>& x) const
  {
    IK_to_EK to_exact;
    csg::Exact_Point_3 p(to_exact(x[0]), to_exact(x[1]), to_exact(x[2]));
    (second_cube * first_cube.inverse())(p);
    values[0] = CGAL::to_double(p[0]);
    values[1] = CGAL::to_double(p[1]);
    values[2] = CGAL::to_double(p[2]);
  }
};

int main() {
  std::string off_file = "../cube.off";
  csg::Exact_Polyhedron_3 cube; // unit cube
  std::cout << "reading file " << off_file << std::endl;
  std::ifstream file(off_file.c_str());
  file >> cube;
  std::cout << "done reading file." << std::endl;
  double cell_size = 1.0;
  bool detect_sharp_features = true;
  Mesh m;
  
  csg::Exact_Polyhedron_3 outer(cube);
  
  // scale the outer box
  CGAL::Aff_transformation_3<csg::Exact_Kernel>
    St(4, 0, 0, 0, 0, 3, 0, 0, 0, 0, 2, 0);
  std::transform(outer.points_begin(), outer.points_end(), 
  		 outer.points_begin(), St);

  // scale the inner box
  CGAL::Aff_transformation_3<csg::Exact_Kernel>
    Et(1, 0, 0, -1.5, 0, 1, 0, 0, 0, 0, 0.2, 0);
  csg::Nef_polyhedron_3 Omega(outer);
  csg::Exact_Polyhedron_3 first_inner(cube);
  std::transform(first_inner.points_begin(), first_inner.points_end(), 
  		 first_inner.points_begin(), Et);

  csg::Exact_Polyhedron_3 second_inner(cube);
  CGAL::Aff_transformation_3<csg::Exact_Kernel> TT = 
    cgal_transformation<csg::Exact_Kernel>(1.0, 1.0, 0.2, 0.0, 0.0, 1.0, 0.3, 1.5, 0.0, 0.0);
  std::transform(second_inner.points_begin(), second_inner.points_end(), 
  		 second_inner.points_begin(), TT);

  Omega -= first_inner;
  Omega -= second_inner;

  csg::Exact_Polyhedron_3 p;
  Omega.convert_to_polyhedron(p);
  csg::Polyhedron_3 q;
  copy_to(p, q);

  generate(m, q, cell_size);
  
  plot(m, "mesh of a cube");
  interactive(true);

  std::cout << "Solving the variational problem" << std::endl;
  model::FunctionSpace V(m);

  CubeDomain first_inner_domain(Et, "first inner");
  CubeToCube first_map(Et, Et);
  CubeDomain second_inner_domain(TT, "second inner");
  CubeToCube second_map(TT, TT);
  CubeDomain outer_domain(St, "outer");
  CubeToCube o2o(St, St);

  // Create Dirichlet boundary conditions
  DirichletBC bci1(V, first_map, first_inner_domain);
  DirichletBC bci2(V, second_map, second_inner_domain);
  DirichletBC bco(V, o2o, outer_domain);
  std::vector<const DirichletBC*> bcs;
  bcs.push_back(&bci1);
  bcs.push_back(&bci2);
  bcs.push_back(&bco);
  
  // Define source and boundary traction functions
  Constant B(0.0, -0.5, 0.0);
  Constant T(0.1,  0.0, 0.0);

  // Define solution function
  Function u(V);

  // Set material parameters
  const double E  = 10.0;
  const double nu = 0.3;
  Constant mu(E/(2*(1 + nu)));
  Constant lambda(E*nu/((1 + nu)*(1 - 2*nu)));

  // Create (linear) form defining (nonlinear) variational problem
  model::ResidualForm F(V);
  F.mu = mu; F.lmbda = lambda; F.B = B; F.T = T; F.u = u;

  // Create jacobian dF = F' (for use in nonlinear solver).
  model::JacobianForm J(V, V);
  J.mu = mu; J.lmbda = lambda; J.u = u;

  // Solve nonlinear variational problem F(u; v) = 0
  solve(F == 0, u, bcs, J);

  // Plot solution
  // plot(u);
  // interactive();

  //std::getchar();

  int N = 5;
  double wx = 7.8/(N-1);
  double wy = 5.8/(N-1);
  double wz = 3.8/(N-1);

  std::fstream out;
  out.open("/home/robertnishihara/function-values.txt", std::ios::out);

  for(int x = 0; x < N; x++) {
    for(int y = 0; y < N; y++) {
      for(int z = 0; z < N; z++) {
	Array<double> values(3);
	Array<double> location(3);
	location[0] = x * wx - 3.9;
	location[1] = y * wy - 2.9;
	location[2] = z * wz - 1.9;

	if(is_inside(TT, location[0], location[1], location[2]) || 
	   is_inside(Et, location[0], location[1], location[2])) {
	  out << "NA" << std::endl;
	  //printf("inside\n");
	}
	else {
	  u.eval(values, location);
	  out << values[0] << " " << values[1] << " " << values[2] << std::endl;
	  //printf("%f %f %f\n", values[0], values[1], values[2]);
	}
      }
    }
  }
  
  out.close();
}
