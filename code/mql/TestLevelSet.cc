#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/compressed_simple_sparsity_pattern.h>
#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/timer.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/base/parameter_handler.h>
#include <fstream>
#include <iostream>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_q_eulerian.h>
#include <deal.II/fe/fe_system.h>

using namespace dealii;

///////////////////////////
// FOR TRANSPORT PROBLEM //
///////////////////////////
// TIME_INTEGRATION
#define FORWARD_EULER 0
#define SSP33 1
// PROBLEM 
#define CIRCULAR_ROTATION 0
#define DIAGONAL_ADVECTION 1
// MESH_VELOCITY (function)
#define ZERO_VELOCITY 0 
#define SINUSOIDAL_WITH_FIXED_BOUNDARY 1
#define SINUSOIDAL_WITH_NON_FIXED_BOUNDARY 2
#define CIRCULAR 3
#define DIAGONAL 4

#include "utilities_test_LS.cc"
#include "LevelSetSolver.cc"

///////////////////////////////////////////////////////
///////////////////// MAIN CLASS //////////////////////
///////////////////////////////////////////////////////
template <int dim>
class TestLevelSet
{
public:
  TestLevelSet (const unsigned int degree_LS,
	    const unsigned int degree_U);
  ~TestLevelSet ();
  void run ();

private:
  // BOUNDARY //
  void set_boundary_inlet();
  void get_boundary_values_phi(MappingQ<dim> &mapping, 
			       std::vector<unsigned int> &boundary_values_id_phi,
			       std::vector<double> &boundary_values_phi);
  // VELOCITY //
  void get_interpolated_velocity(Mapping<dim> &mapping);
  // SETUP AND INIT CONDITIONS //
  void setup();
  void initial_condition(Mapping<dim> &mapping);
  void init_constraints();
  // POST PROCESSING //
  void process_solution(Mapping<dim> &mapping,
			parallel::distributed::Triangulation<dim> &triangulation,
			DoFHandler<dim> &dof_handler_LS, 
			PETScWrappers::MPI::Vector &solution);
  void output_results(Mapping<dim> &mapping);
  void output_solution(Mapping<dim> &mapping);
  void output_rho(Mapping<dim> &mapping);
  // MOVING MESH //
  double get_time_step(Mapping<dim> &mapping, Vector<double> coord);
  void get_interpolated_mesh_velocity(Mapping<dim> &mapping);
  void compute_vertex_coord(Vector<double> &old_coord);
  void compute_mesh_displacement(double dt, 
				 PETScWrappers::MPI::Vector &mesh_vel_u, 
				 PETScWrappers::MPI::Vector &mesh_vel_v, 
				 Vector<double> &old_coord,
				 Vector<double> &new_coord, 
				 Vector<double> &mesh_disp);
  // MOVING MESH VARIABLES //
  unsigned int MESH_VELOCITY;
  PETScWrappers::MPI::Vector locally_relevant_mesh_velocity_u;
  PETScWrappers::MPI::Vector locally_relevant_mesh_velocity_v;
  PETScWrappers::MPI::Vector completely_distributed_mesh_velocity_u;
  PETScWrappers::MPI::Vector completely_distributed_mesh_velocity_v;

  // SOLUTION VECTORS
  PETScWrappers::MPI::Vector locally_relevant_solution_phi;
  PETScWrappers::MPI::Vector locally_relevant_solution_u;
  PETScWrappers::MPI::Vector locally_relevant_solution_v;
  PETScWrappers::MPI::Vector completely_distributed_solution_phi;
  PETScWrappers::MPI::Vector completely_distributed_solution_u;
  PETScWrappers::MPI::Vector completely_distributed_solution_v;
  // BOUNDARY VECTORS
  std::vector<unsigned int> boundary_values_id_phi;
  std::vector<double> boundary_values_phi;

  // GENERAL 
  MPI_Comm mpi_communicator;
  parallel::distributed::Triangulation<dim>   triangulation;
  
  int                  degree;
  int                  degree_LS;
  DoFHandler<dim>      dof_handler_LS;
  FE_Q<dim>            fe_LS;
  IndexSet             locally_owned_dofs_LS;
  IndexSet             locally_relevant_dofs_LS;

  int                  degree_U;
  DoFHandler<dim>      dof_handler_U;
  FE_Q<dim>            fe_U;
  IndexSet             locally_owned_dofs_U;
  IndexSet             locally_relevant_dofs_U;

  DoFHandler<dim>      dof_handler_U_disp_field;
  FESystem<dim>        fe_U_disp_field;
  IndexSet             locally_owned_dofs_U_disp_field;
  IndexSet             locally_relevant_dofs_U_disp_field;

  ConstraintMatrix     constraints;
  ConstraintMatrix     constraints_disp_field;

  double time;
  double time_step;
  double final_time;
  unsigned int timestep_number;
  double cfl;
  double min_h;

  double sharpness; 
  int sharpness_integer;

  unsigned int n_refinement;
  unsigned int output_number;
  double output_time;
  bool get_output;

  bool verbose;
  ConditionalOStream pcout;

  //FOR TRANSPORT
  double cK; //compression coeff
  double cE; //entropy-visc coeff
  unsigned int TRANSPORT_TIME_INTEGRATION;
  std::string ALGORITHM;
  unsigned int PROBLEM;

  //FOR RECONSTRUCTION OF MATERIAL FIELDS
  double eps, rho_air, rho_fluid;

  // MASS MATRIX
  PETScWrappers::MPI::SparseMatrix matrix_MC, matrix_MC_tnm1;
  std_cxx1x::shared_ptr<PETScWrappers::PreconditionBoomerAMG> preconditioner_MC;
  
};

template <int dim>
TestLevelSet<dim>::TestLevelSet (const unsigned int degree_LS, 
			 const unsigned int degree_U)
  :
  mpi_communicator (MPI_COMM_WORLD),
  triangulation (mpi_communicator,
		 typename Triangulation<dim>::MeshSmoothing
		 (Triangulation<dim>::smoothing_on_refinement |
		  Triangulation<dim>::smoothing_on_coarsening)),
  degree_LS(degree_LS),
  dof_handler_LS (triangulation),
  fe_LS (degree_LS),
  degree_U(degree_U),
  dof_handler_U (triangulation),
  fe_U (degree_U),
  dof_handler_U_disp_field(triangulation),
  fe_U_disp_field(FE_Q<dim>(degree_U),dim),
  pcout (std::cout,(Utilities::MPI::this_mpi_process(mpi_communicator)== 0))
{}

template <int dim>
TestLevelSet<dim>::~TestLevelSet ()
{
  dof_handler_U_disp_field.clear();
  dof_handler_LS.clear ();
  dof_handler_U.clear ();
}

// VELOCITY //
//////////////
template <int dim>
void TestLevelSet<dim>::get_interpolated_velocity(Mapping<dim> &mapping)
{
  // velocity in x
  completely_distributed_solution_u = 0;
  VectorTools::interpolate(mapping,dof_handler_U,
			   ExactU<dim>(PROBLEM,time),
			   completely_distributed_solution_u);
  constraints.distribute (completely_distributed_solution_u);
  locally_relevant_solution_u = completely_distributed_solution_u;
  // velocity in y
  completely_distributed_solution_v = 0;
  VectorTools::interpolate(mapping,dof_handler_U,
			   ExactV<dim>(PROBLEM,time),
			   completely_distributed_solution_v);
  constraints.distribute (completely_distributed_solution_v);
  locally_relevant_solution_v = completely_distributed_solution_v;
}

//////////////
// BOUNDARY //
//////////////
template <int dim>
void TestLevelSet<dim>::set_boundary_inlet()
{
  const QGauss<dim-1>  face_quadrature_formula(1); // center of the face
  FEFaceValues<dim> fe_face_values (fe_U,face_quadrature_formula,
				    update_values | update_quadrature_points |
				    update_normal_vectors);
  const unsigned int n_face_q_points = face_quadrature_formula.size();
  std::vector<double>  u_value (n_face_q_points);
  std::vector<double>  v_value (n_face_q_points); 
  
  typename DoFHandler<dim>::active_cell_iterator
    cell_U = dof_handler_U.begin_active(),
    endc_U = dof_handler_U.end();
  Tensor<1,dim> u;
  
  for (; cell_U!=endc_U; ++cell_U)
    if (cell_U->is_locally_owned())
      for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
	if (cell_U->face(face)->at_boundary())
	  {
	    fe_face_values.reinit(cell_U,face);
	    fe_face_values.get_function_values(locally_relevant_solution_u,u_value);
	    fe_face_values.get_function_values(locally_relevant_solution_v,v_value);
	    u[0]=u_value[0];
	    u[1]=v_value[0];
	    if (fe_face_values.normal_vector(0)*u < -1e-14)
	      cell_U->face(face)->set_boundary_id(10);
	  }
}

template <int dim>
void TestLevelSet<dim>::get_boundary_values_phi(MappingQ<dim> &mapping, 
						std::vector<unsigned int> &boundary_values_id_phi,
						std::vector<double> &boundary_values_phi)
{
  std::map<unsigned int, double> map_boundary_values_phi;
  unsigned int boundary_id=0;
  
  set_boundary_inlet();
  boundary_id=10; // inlet
  VectorTools::interpolate_boundary_values (mapping,dof_handler_LS,
					    boundary_id,BoundaryPhi<dim>(),
					    map_boundary_values_phi);

  boundary_values_id_phi.resize(map_boundary_values_phi.size());
  boundary_values_phi.resize(map_boundary_values_phi.size());  
  std::map<unsigned int,double>::const_iterator boundary_value_phi = map_boundary_values_phi.begin();
  for (int i=0; boundary_value_phi !=map_boundary_values_phi.end(); ++boundary_value_phi, ++i)
    {
      boundary_values_id_phi[i]=boundary_value_phi->first;
      boundary_values_phi[i]=boundary_value_phi->second;
    }
}

///////////////////////////////////
// SETUP AND INITIAL CONDITIONS //
//////////////////////////////////
template <int dim>
void TestLevelSet<dim>::setup()
{ 
  degree = std::max(degree_LS,degree_U);
  // setup system LS
  dof_handler_LS.distribute_dofs (fe_LS);
  locally_owned_dofs_LS = dof_handler_LS.locally_owned_dofs ();
  DoFTools::extract_locally_relevant_dofs (dof_handler_LS,
					   locally_relevant_dofs_LS);
  // setup system U 
  dof_handler_U.distribute_dofs (fe_U);
  locally_owned_dofs_U = dof_handler_U.locally_owned_dofs ();
  DoFTools::extract_locally_relevant_dofs (dof_handler_U,
					   locally_relevant_dofs_U);
  // setup system U for disp field
  dof_handler_U_disp_field.distribute_dofs (fe_U_disp_field);
  locally_owned_dofs_U_disp_field = dof_handler_U_disp_field.locally_owned_dofs ();
  DoFTools::extract_locally_relevant_dofs (dof_handler_U_disp_field,
					   locally_relevant_dofs_U_disp_field);
  // init vectors for phi
  locally_relevant_solution_phi.reinit(locally_owned_dofs_LS,
				       locally_relevant_dofs_LS,
				       mpi_communicator);
  locally_relevant_solution_phi = 0;
  completely_distributed_solution_phi.reinit(mpi_communicator, 
					     dof_handler_LS.n_dofs(),
					     dof_handler_LS.n_locally_owned_dofs());
  //init vectors for u
  locally_relevant_solution_u.reinit(locally_owned_dofs_U,
				     locally_relevant_dofs_U,
				     mpi_communicator);
  locally_relevant_solution_u = 0;
  completely_distributed_solution_u.reinit(mpi_communicator, 
					   dof_handler_U.n_dofs(),
					   dof_handler_U.n_locally_owned_dofs());
  //init vectors for v                                           
  locally_relevant_solution_v.reinit(locally_owned_dofs_U,
				     locally_relevant_dofs_U,
				     mpi_communicator);
  locally_relevant_solution_v = 0;
  completely_distributed_solution_v.reinit(mpi_communicator, 
					   dof_handler_U.n_dofs(),
					   dof_handler_U.n_locally_owned_dofs());
  // MOVING MESH
  locally_relevant_mesh_velocity_u.reinit(locally_owned_dofs_U,locally_relevant_dofs_U,mpi_communicator); 
  locally_relevant_mesh_velocity_v.reinit(locally_owned_dofs_U,locally_relevant_dofs_U,mpi_communicator); 
  completely_distributed_mesh_velocity_u.reinit(locally_owned_dofs_U,mpi_communicator); 
  completely_distributed_mesh_velocity_v.reinit(locally_owned_dofs_U,mpi_communicator); 
  init_constraints();
  // MASS MATRIX
  DynamicSparsityPattern dsp (locally_relevant_dofs_LS);
  DoFTools::make_sparsity_pattern (dof_handler_LS,dsp,constraints,false);
  SparsityTools::distribute_sparsity_pattern (dsp,
					      dof_handler_LS.n_locally_owned_dofs_per_processor(),
					      mpi_communicator,
					      locally_relevant_dofs_LS);
  matrix_MC.reinit (mpi_communicator,
		    dsp,
		    dof_handler_LS.n_locally_owned_dofs_per_processor(),
		    dof_handler_LS.n_locally_owned_dofs_per_processor(),
		    Utilities::MPI::this_mpi_process(mpi_communicator));
  matrix_MC_tnm1.reinit (mpi_communicator,
			 dsp,
			 dof_handler_LS.n_locally_owned_dofs_per_processor(),
			 dof_handler_LS.n_locally_owned_dofs_per_processor(),
			 Utilities::MPI::this_mpi_process(mpi_communicator));
}

template <int dim>
void TestLevelSet<dim>::initial_condition(Mapping<dim> &mapping)
{
  time=0;
  // Initial conditions //
  // init condition for phi
  completely_distributed_solution_phi = 0;
  VectorTools::interpolate(mapping,dof_handler_LS,
			   InitialPhi<dim>(PROBLEM, sharpness),
			   //ZeroFunction<dim>(),
			   completely_distributed_solution_phi);
  constraints.distribute (completely_distributed_solution_phi);
  locally_relevant_solution_phi = completely_distributed_solution_phi;
  // init condition for u=0
  completely_distributed_solution_u = 0;
  VectorTools::interpolate(mapping,dof_handler_U,
			   ExactU<dim>(PROBLEM,time),
			   completely_distributed_solution_u);
  constraints.distribute (completely_distributed_solution_u);
  locally_relevant_solution_u = completely_distributed_solution_u;
  // init condition for v
  completely_distributed_solution_v = 0;
  VectorTools::interpolate(mapping,dof_handler_U,
			   ExactV<dim>(PROBLEM,time),
			   completely_distributed_solution_v);
  constraints.distribute (completely_distributed_solution_v);
  locally_relevant_solution_v = completely_distributed_solution_v;
}
  
template <int dim>
void TestLevelSet<dim>::init_constraints()
{
  constraints.clear ();
  constraints.reinit (locally_relevant_dofs_LS);
  DoFTools::make_hanging_node_constraints (dof_handler_LS, constraints);
  constraints.close ();
  constraints_disp_field.clear ();
  constraints_disp_field.reinit (locally_relevant_dofs_LS);
  DoFTools::make_hanging_node_constraints (dof_handler_LS, constraints_disp_field);
  constraints_disp_field.close ();
}

/////////////////////
// POST PROCESSING //
/////////////////////
template <int dim>
void TestLevelSet<dim>::process_solution(Mapping<dim> &mapping, 
				     parallel::distributed::Triangulation<dim> &triangulation, 
				     DoFHandler<dim> &dof_handler_LS, 
				     PETScWrappers::MPI::Vector &solution)
{
  Vector<double> difference_per_cell (triangulation.n_active_cells());
  // error for phi
  VectorTools::integrate_difference (mapping,dof_handler_LS,
				     solution,
				     InitialPhi<dim>(PROBLEM,sharpness),
				     difference_per_cell,
				     QGauss<dim>(degree_LS+3),
				     VectorTools::L1_norm);
  
  double u_L1_error = difference_per_cell.l1_norm();
  u_L1_error = std::sqrt(Utilities::MPI::sum(u_L1_error * u_L1_error, mpi_communicator));
  
  VectorTools::integrate_difference (mapping,dof_handler_LS,
				     solution,
				     InitialPhi<dim>(PROBLEM,sharpness),
				     difference_per_cell,
				     QGauss<dim>(degree_LS+3),
				     VectorTools::L2_norm);
  double u_L2_error = difference_per_cell.l2_norm();
  u_L2_error = std::sqrt(Utilities::MPI::sum(u_L2_error * u_L2_error, mpi_communicator));
  
  pcout << "L1 error: " << u_L1_error << std::endl;
  pcout << "L2 error: " << u_L2_error << std::endl;
}

template<int dim>
void TestLevelSet<dim>::output_results(Mapping<dim> &mapping)
{
  output_solution(mapping);
  //output_rho(mapping);
  output_number++;
}

template <int dim>
void TestLevelSet<dim>::output_solution(Mapping<dim> &mapping)
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler_LS);  
  data_out.add_data_vector (locally_relevant_solution_phi, "phi");
  data_out.build_patches(mapping);

  const std::string filename = ("solution-" +
				Utilities::int_to_string (output_number, 3) +
				"." +
				Utilities::int_to_string
				(triangulation.locally_owned_subdomain(), 4));
  std::ofstream output ((filename + ".vtu").c_str());
  data_out.write_vtu (output);
  
  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      std::vector<std::string> filenames;
      for (unsigned int i=0;
	   i<Utilities::MPI::n_mpi_processes(mpi_communicator);
	   ++i)
	filenames.push_back ("solution-" +
			     Utilities::int_to_string (output_number, 3) +
			     "." +
			     Utilities::int_to_string (i, 4) +
			     ".vtu");
      
      std::ofstream master_output ((filename + ".pvtu").c_str());
      data_out.write_pvtu_record (master_output, filenames);
    }
}

template <int dim>
void TestLevelSet<dim>::output_rho(Mapping<dim> &mapping)
{
  Postprocessor<dim> postprocessor(eps,rho_air,rho_fluid);
  DataOut<dim> data_out;
  data_out.attach_dof_handler (dof_handler_LS);  
  data_out.add_data_vector (locally_relevant_solution_phi, postprocessor);
  data_out.build_patches (mapping,degree_LS);
  
  const std::string filename = ("sol_rho-" +
				Utilities::int_to_string (output_number, 3) +
				"." +
				Utilities::int_to_string
				(triangulation.locally_owned_subdomain(), 4));
  std::ofstream output ((filename + ".vtu").c_str());
  data_out.write_vtu (output);
  
  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      std::vector<std::string> filenames;
      for (unsigned int i=0;
	   i<Utilities::MPI::n_mpi_processes(mpi_communicator);
	   ++i)
	filenames.push_back ("sol_rho-" +
			     Utilities::int_to_string (output_number, 3) +
			     "." +
			     Utilities::int_to_string (i, 4) +
			     ".vtu");
      
      std::ofstream master_output ((filename + ".pvtu").c_str());
      data_out.write_pvtu_record (master_output, filenames);
    }
}

/////////////////
// MOVING MESH //
/////////////////
template<int dim>
double TestLevelSet<dim>::get_time_step(Mapping<dim> &mapping, Vector<double> coord)
{  
  Vector<double> umax_per_cell (triangulation.n_active_cells());
  VectorTools::integrate_difference (mapping,dof_handler_U,
				     locally_relevant_solution_u,
				     ZeroFunction<dim>(),
				     umax_per_cell,
				     QGauss<dim>(degree+1),
				     VectorTools::Linfty_norm);
  Vector<double> vmax_per_cell (triangulation.n_active_cells());
  VectorTools::integrate_difference (mapping,dof_handler_U,
				     locally_relevant_solution_v,
				     ZeroFunction<dim>(),
				     vmax_per_cell,
				     QGauss<dim>(degree+1),
				     VectorTools::Linfty_norm);
  // get dt
  const unsigned int   dofs_per_cell = fe_U_disp_field.dofs_per_cell;
  std::vector<unsigned int> local_dof_indices (dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator
    cell_LS = dof_handler_U_disp_field.begin_active(),
    endc_LS = dof_handler_U_disp_field.end();
  double xi, yi, xj, yj, dist, h;
  int cell=0;
  double dt=1000;
  for (; cell_LS!=endc_LS; ++cell_LS, ++cell)
    if (cell_LS->is_locally_owned())
      {
  	cell_LS->get_dof_indices (local_dof_indices);
	h=1000;
	for (unsigned int i=0; i<GeometryInfo<dim>::vertices_per_cell; ++i)
	  {
	    xi = coord(local_dof_indices[fe_U_disp_field.component_to_system_index(0,i)]);
	    yi = coord(local_dof_indices[fe_U_disp_field.component_to_system_index(1,i)]);
	    for (unsigned int j=0; j<GeometryInfo<dim>::vertices_per_cell; ++j)
	      if(j!=i)
		{
		  xj = coord(local_dof_indices[fe_U_disp_field.component_to_system_index(0,j)]);
		  yj = coord(local_dof_indices[fe_U_disp_field.component_to_system_index(1,j)]);
		  dist = std::sqrt(std::pow(xi-xj,2) + std::pow(yi-yj,2));
		  h = std::min(h,dist);
		}
	  }
	double UMAX = std::sqrt(std::pow(umax_per_cell(cell),2)+std::pow(vmax_per_cell(cell),2));
	dt = std::min(dt,cfl*h/UMAX);
      }
  return Utilities::MPI::min(dt,mpi_communicator);
}

template<int dim>
void TestLevelSet<dim>::get_interpolated_mesh_velocity(Mapping<dim> &mapping)
{
  completely_distributed_mesh_velocity_u = 0;
  VectorTools::interpolate(mapping,dof_handler_U,
			   MeshVelocityU<dim>(MESH_VELOCITY,time),
			   completely_distributed_mesh_velocity_u);
  constraints.distribute (completely_distributed_mesh_velocity_u);
  locally_relevant_mesh_velocity_u=completely_distributed_mesh_velocity_u;
  completely_distributed_mesh_velocity_v = 0;
  VectorTools::interpolate(mapping,dof_handler_U,
			   MeshVelocityV<dim>(MESH_VELOCITY,time),
			   completely_distributed_mesh_velocity_v);
  constraints.distribute (completely_distributed_mesh_velocity_v);  
  locally_relevant_mesh_velocity_v=completely_distributed_mesh_velocity_v;
}

template<int dim>
void TestLevelSet<dim>::compute_vertex_coord(Vector<double> &old_coord)
{
  old_coord = 0;
  const unsigned int   dofs_per_cell = fe_U_disp_field.dofs_per_cell;
  std::vector<unsigned int> local_dof_indices (dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator
    cell_LS = dof_handler_U_disp_field.begin_active(),
    endc_LS = dof_handler_U_disp_field.end();

  for (; cell_LS!=endc_LS; ++cell_LS)
    if (cell_LS->is_locally_owned())
      {
  	cell_LS->get_dof_indices (local_dof_indices);
	for (unsigned int i=0; i<GeometryInfo<dim>::vertices_per_cell; ++i)
	  {
	    old_coord(local_dof_indices[fe_U_disp_field.component_to_system_index(0,i)]) 
	      = cell_LS->vertex(i)[0];
	    old_coord(local_dof_indices[fe_U_disp_field.component_to_system_index(1,i)]) 
	      = cell_LS->vertex(i)[1];
	  }
      }
}

template<int dim>
void TestLevelSet<dim>::compute_mesh_displacement(double dt, 
						  PETScWrappers::MPI::Vector &mesh_vel_u, 
						  PETScWrappers::MPI::Vector &mesh_vel_v, 
						  Vector<double> &old_coord,
						  Vector<double> &new_coord,
						  Vector<double> &mesh_disp)
{
  const unsigned int   dofs_per_cell = fe_U_disp_field.dofs_per_cell;
  std::vector<unsigned int> local_dof_indices (dofs_per_cell);
  std::vector<unsigned int> local_dof_indices_U_uncoupled (fe_U.dofs_per_cell); //fe_U is Q1

  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler_U_disp_field.begin_active(),
    endc = dof_handler_U_disp_field.end();
  typename DoFHandler<dim>::active_cell_iterator cell_U_uncoupled = dof_handler_U.begin_active();

  double vx,vy,x,y;
  double DispX, DispY, new_coordX, new_coordY;
  for (; cell!=endc; ++cell,++cell_U_uncoupled)
    if (cell->is_locally_owned())
      {
  	cell->get_dof_indices (local_dof_indices);
  	cell_U_uncoupled->get_dof_indices (local_dof_indices_U_uncoupled);
	for (unsigned int i=0; i<GeometryInfo<dim>::vertices_per_cell; ++i)
	  {
	    x = old_coord(local_dof_indices[fe_U_disp_field.component_to_system_index(0,i)]);
	    y = old_coord(local_dof_indices[fe_U_disp_field.component_to_system_index(1,i)]);
	    vx = mesh_vel_u(local_dof_indices_U_uncoupled[i]);
	    vy = mesh_vel_v(local_dof_indices_U_uncoupled[i]);
	    // compute new coord: old coord+displacement_on_time_step
	    // compute new coord
	    new_coordX = x+vx*dt;
	    new_coordY = y+vy*dt;
	    new_coord(local_dof_indices[fe_U_disp_field.component_to_system_index(0,i)]) = new_coordX;
	    new_coord(local_dof_indices[fe_U_disp_field.component_to_system_index(1,i)]) = new_coordY;
	    // compute global displacement; i.e., from original coord
	    DispX = new_coordX - cell->vertex(i)[0]; //new_coord - original_coord
	    DispY = new_coordY - cell->vertex(i)[1];

	    mesh_disp(local_dof_indices[fe_U_disp_field.component_to_system_index(0,i)])=DispX;
	    mesh_disp(local_dof_indices[fe_U_disp_field.component_to_system_index(1,i)])=DispY;
	  }
      }
}

template <int dim>
void TestLevelSet<dim>::run()
{
  ////////////////////////
  // GENERAL PARAMETERS //
  ////////////////////////
  cfl=0.1;
  verbose = true;
  get_output = true;
  output_number = 0;
  Timer t;
  n_refinement=6;
  output_time = 0.1;
  final_time = 0.5;
  //PROBLEM=CIRCULAR_ROTATION;
  PROBLEM=DIAGONAL_ADVECTION;

  //////////////////////////////////////
  // PARAMETERS FOR TRANSPORT PROBLEM //
  //////////////////////////////////////
  cK = 1.0; // compression constant
  cE = 1.0; // entropy viscosity constant
  sharpness_integer=1; //this will be multipled by min_h
  TRANSPORT_TIME_INTEGRATION=FORWARD_EULER;
  //TRANSPORT_TIME_INTEGRATION=SSP33;
  //ALGORITHM = "MPP_u1";
  ALGORITHM = "NMPP_uH";
  //ALGORITHM = "MPP_uH";
  
  //////////////
  // GEOMETRY //
  //////////////
  if (PROBLEM==CIRCULAR_ROTATION || PROBLEM==DIAGONAL_ADVECTION) 
    GridGenerator::hyper_rectangle(triangulation, Point<dim>(0.0,0.0), Point<dim>(1.0,1.0), true);      
  triangulation.refine_global (n_refinement);
  
  ///////////
  // SETUP //
  ///////////
  setup();

  // for Reconstruction of MATERIAL FIELDS
  min_h = GridTools::minimal_cell_diameter(triangulation)/std::sqrt(2)/degree;
  eps=1*min_h; //For reconstruction of density in Navier Stokes
  sharpness=sharpness_integer*min_h; //adjust value of sharpness (for init cond of phi)
  rho_fluid = 1000;
  rho_air = 1;

  /////////////
  // MAPPING //
  /////////////
  //MappingQ<dim> mapping_tnm1(1);
  //MappingQ<dim> mapping_tn(1);
  Vector<double> mesh_disp_tn(dof_handler_U_disp_field.n_dofs()); mesh_disp_tn = 0;
  Vector<double> mesh_disp_tnm1(dof_handler_U_disp_field.n_dofs()); mesh_disp_tnm1 = 0;
  Vector<double> old_coord(dof_handler_U_disp_field.n_dofs()); old_coord = 0;
  Vector<double> new_coord(dof_handler_U_disp_field.n_dofs()); new_coord = 0;
  MappingQEulerian<dim,Vector<double>,dim> mapping_tn(1,dof_handler_U_disp_field,mesh_disp_tn);
  MappingQEulerian<dim,Vector<double>,dim> mapping_tnm1(1,dof_handler_U_disp_field,mesh_disp_tnm1);

  //////////////////////
  // TRANSPORT SOLVER //
  //////////////////////
  LevelSetSolver<dim> level_set (degree_LS,degree_U,
				 1.0,cK,cE, // dummy initial time_step=1
				 verbose, 
				 ALGORITHM,
				 TRANSPORT_TIME_INTEGRATION,
				 triangulation,
				 mapping_tn,
				 mapping_tnm1,
				 mpi_communicator); 
  ///////////////////////
  // INITIAL CONDITION //
  ///////////////////////
  initial_condition(mapping_tn);
  output_results(mapping_tn);
  level_set.initial_condition(locally_relevant_solution_phi,
			      locally_relevant_solution_u,locally_relevant_solution_v);
  
  /////////////////
  // MOVING MESH //
  /////////////////
  MESH_VELOCITY = SINUSOIDAL_WITH_FIXED_BOUNDARY;
  //MESH_VELOCITY = ZERO_VELOCITY;
  compute_vertex_coord(old_coord); // initial location of vertices
  
  /////////////////////////////////
  // BOUNDARY CONDITIONS FOR PHI // 
  /////////////////////////////////
  get_boundary_values_phi(mapping_tn,boundary_values_id_phi,boundary_values_phi);
  level_set.set_boundary_conditions(boundary_values_id_phi,boundary_values_phi);
    
  // OUTPUT DATA REGARDING TIME STEPPING AND MESH //
  int dofs_LS = dof_handler_LS.n_dofs();
  pcout << "Cfl: " << cfl << std::endl;
  pcout << "   Number of active cells:       " 
  	<< triangulation.n_global_active_cells() << std::endl
  	<< "   Number of degrees of freedom: " << std::endl
  	<< "      LS: " << dofs_LS << std::endl;

  // TIME STEPPING	
  timestep_number=0;
  time=0;
  while(time<final_time)
    { 
      timestep_number++;
      ///////////////////
      // GET TIME_STEP // get dt for current time tn
      ///////////////////
      time_step = get_time_step(mapping_tn,old_coord);
      if (time+time_step > final_time)
	{ 
	  pcout << "FINAL TIME STEP... " << std::endl; 
	  time_step = final_time-time;
	}
      pcout << "Time step " << timestep_number 
	    << "\twith dt=" << time_step 
	    << "\tat tn=" << time << std::endl;

      ///////////////////////
      // GET MESH VELOCITY // at current time tn
      ///////////////////////
      get_interpolated_mesh_velocity(mapping_tn);
      level_set.set_mesh_velocity(locally_relevant_mesh_velocity_u, locally_relevant_mesh_velocity_v);

      /////////////////////////////
      // GET (RELATIVE) VELOCITY // (NS or interpolate from a function) at current time tn
      /////////////////////////////
      get_interpolated_velocity(mapping_tn);
      // ADJUST VELOCITY (Physical velocity - mesh velocity)
      completely_distributed_solution_u.add(-1.0,completely_distributed_mesh_velocity_u);
      completely_distributed_solution_v.add(-1.0,completely_distributed_mesh_velocity_v);
      locally_relevant_solution_u=completely_distributed_solution_u;
      locally_relevant_solution_v=completely_distributed_solution_v;
      // SET VELOCITY TO LEVEL SET SOLVER
      level_set.set_velocity(locally_relevant_solution_u,locally_relevant_solution_v);
      
      ////////////////////////////
      // GET LEVEL SET SOLUTION // (at tnp1)
      ////////////////////////////
      level_set.nth_time_step(time_step);

      /////////////////
      // UPDATE TIME //
      /////////////////
      time+=time_step; // time tnp1

      ///////////////
      // MOVE MESH // compute mapping tn (for next time step)
      ///////////////
      mesh_disp_tnm1=mesh_disp_tn;
      compute_mesh_displacement(time_step,
				locally_relevant_mesh_velocity_u,
				locally_relevant_mesh_velocity_v,
				old_coord,new_coord,mesh_disp_tn);
      old_coord.equ(1.0,new_coord); // update old coordinates

      ////////////
      // OUTPUT //
      ////////////
      if (get_output && time-(output_number)*output_time>=0)
	{
	  level_set.get_unp1(locally_relevant_solution_phi);
	  output_results(mapping_tn);
	}
    }
  pcout << "FINAL TIME T=" << time << std::endl;
}

int main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);
      deallog.depth_console (0);
      {
	unsigned int degree = 1;
        TestLevelSet<2> multiphase(degree, degree);
        multiphase.run();
      }
      PetscFinalize();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  return 0;
}
