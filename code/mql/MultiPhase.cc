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

#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <fstream>

#include <numeric>      // std::accumulate

using namespace dealii;

///////////////////////////
// FOR TRANSPORT PROBLEM //
///////////////////////////
// TIME_INTEGRATION
#define FORWARD_EULER 1
#define SSP33 0
// PROBLEM 
#define SMALL_WAVE_PERTURBATION 0
#define BOX_PROBLEM 1 
#define FLOATING 2
// MESH VELOCITY 
#define ZERO_VELOCITY 0 
#define SINUSOIDAL_WITH_FIXED_BOUNDARY 1
#define BOX_VELOCITY 2

// OTHERS 
#define NUM_ITER_TO_RECOMPUTE_PRECONDITIONER 10

#include "NavierStokesSolver_BDF1.cc"
#include "LevelSetSolver.cc"
#include "utilities.cc"

///////////////////////////////////////////////////////
///////////////////// MAIN CLASS //////////////////////
///////////////////////////////////////////////////////
template <int dim>
class MultiPhase
{
public:
  MultiPhase (const unsigned int degree_LS,
	      const unsigned int degree_U);
  ~MultiPhase ();
  void run ();

private:
  void set_boundary_inlet(MappingQ<dim> &mapping);
  void get_boundary_values_U(MappingQ<dim> &mapping, double time);
  void get_boundary_values_phi(MappingQ<dim> &mapping, 
			       std::vector<unsigned int> &boundary_values_id_phi,
			       std::vector<double> &boundary_values_phi);
  void output_results(MappingQ<dim> &mapping);
  void output_vectors(MappingQ<dim> &mapping);
  void output_rho(MappingQ<dim> &mapping);
  void setup();
  void initial_condition(MappingQ<dim> &mapping);
  void init_constraints();
  // MOVING MESH //
  void interpolate_from_Q1_to_Q2(PETScWrappers::MPI::Vector &vector_in_Q1,
				 PETScWrappers::MPI::Vector &vector_in_Q2);				 
  void get_sparsity_pattern();
  void get_boundary_map();
  void get_global_Q1_to_Q2_map();
  void get_maps_at_box();
  int solve(const ConstraintMatrix &constraints, 
	    PETScWrappers::MPI::SparseMatrix &Matrix,
	    std_cxx1x::shared_ptr<PETScWrappers::PreconditionBoomerAMG> preconditioner,
	    PETScWrappers::MPI::Vector &completely_distributed_solution,
	    const PETScWrappers::MPI::Vector &rhs);
  void get_smoothed_velocity_via_laplacian(MappingQ<dim> &mapping,
					   PETScWrappers::MPI::Vector &velocity_to_smooth_u,
					   PETScWrappers::MPI::Vector &velocity_to_smooth_v,
					   PETScWrappers::MPI::Vector &smoothed_velocity_u,
					   PETScWrappers::MPI::Vector &smoothed_velocity_v);
  void get_smoothed_velocity_via_gravity_center(double dt, Vector<double> &old_coord,
						PETScWrappers::MPI::Vector &velocity_to_smooth_u,
						PETScWrappers::MPI::Vector &velocity_to_smooth_v,
						PETScWrappers::MPI::Vector &smoothed_velocity_u,
						PETScWrappers::MPI::Vector &smoothed_velocity_v);
  double get_time_step(MappingQ<dim> &mapping, Vector<double> coord);
  void get_interpolated_mesh_velocity(MappingQ<dim> &mapping);
  void compute_vertex_coord(Vector<double> &old_coord);
  void compute_mesh_displacement_via_vectors_in_Q1(double dt, 
						   PETScWrappers::MPI::Vector &mesh_vel_u,
						   PETScWrappers::MPI::Vector &mesh_vel_v,
						   Vector<double> &old_coord,
						   Vector<double> &new_coord, 
						   Vector<double> &mesh_disp);
  void compute_mesh_displacement_via_vectors_in_Q2(double dt, 
						   PETScWrappers::MPI::Vector &mesh_vel_u,
						   PETScWrappers::MPI::Vector &mesh_vel_v,
						   Vector<double> &old_coord,
						   Vector<double> &new_coord, 
						   Vector<double> &mesh_disp);
  void compute_new_coordinates_from_mesh_velocity_vectors(double dt, 
							  PETScWrappers::MPI::Vector &mesh_vel_u,
							  PETScWrappers::MPI::Vector &mesh_vel_v,
							  Vector<double> &old_coord,
							  PETScWrappers::MPI::Vector &x_old_coord,
							  PETScWrappers::MPI::Vector &y_old_coord,
							  PETScWrappers::MPI::Vector &x_new_coord,
							  PETScWrappers::MPI::Vector &y_new_coord);
  void make_structured_box(parallel::distributed::Triangulation <dim> &tria);
  void get_mu(double &mu, double phi);
  void get_body_velocity(MappingQ<dim> &mapping, Point<dim> &XG, Vector<double> &VG, double &thetaG, double &omegaG,
			 double time_step, 
			 PETScWrappers::MPI::Vector &fluid_velocity_u_Q2,
			 PETScWrappers::MPI::Vector &fluid_velocity_v_Q2,
			 PETScWrappers::MPI::Vector &body_velocity_u_Q1,
			 PETScWrappers::MPI::Vector &body_velocity_v_Q1);

  MPI_Comm mpi_communicator;
  parallel::distributed::Triangulation<dim>   triangulation;
  
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

  DoFHandler<dim>      dof_handler_P;
  FE_Q<dim>            fe_P;
  IndexSet             locally_owned_dofs_P;
  IndexSet             locally_relevant_dofs_P;

  DoFHandler<dim>      dof_handler_U_mesh;
  FE_Q<dim>            fe_U_mesh;
  IndexSet             locally_owned_dofs_U_mesh;
  IndexSet             locally_relevant_dofs_U_mesh;
  
  DoFHandler<dim>      dof_handler_U_disp_field;
  FESystem<dim>        fe_U_disp_field;
  IndexSet             locally_owned_dofs_U_disp_field;
  IndexSet             locally_relevant_dofs_U_disp_field;

  ConstraintMatrix     constraints_Q1;
  ConstraintMatrix     constraints_Q2;
  ConstraintMatrix     constraints_disp_field;

  ConditionalOStream                pcout;

  std::map<types::global_dof_index, std::vector<types::global_dof_index> > sparsity_pattern;
  std::map<types::global_dof_index, bool> is_dof_at_boundary;
  std::map<typename DoFHandler<dim>::active_cell_iterator, bool> cell_U_mesh_at_box;
  std::map<types::global_dof_index, bool> is_dof_at_box;
  std::map<types::global_dof_index, types::global_dof_index> global_Q1_to_Q2_map;

  // MOVING MESH VARIABLES //
  Vector<double> x_old_coord;
  Vector<double> y_old_coord;
  Vector<double> x_new_coord;
  Vector<double> y_new_coord;
  // SMOOTHING MESH 
  PETScWrappers::MPI::Vector locally_relevant_smoothed_lm1_x_new_coord;
  PETScWrappers::MPI::Vector locally_relevant_smoothed_lm1_y_new_coord;
  PETScWrappers::MPI::Vector completely_distributed_smoothed_x_new_coord;
  PETScWrappers::MPI::Vector completely_distributed_smoothed_y_new_coord;
  PETScWrappers::MPI::Vector completely_distributed_non_smooth_x_new_coord;
  PETScWrappers::MPI::Vector completely_distributed_non_smooth_y_new_coord;
  PETScWrappers::MPI::Vector completely_distributed_x_old_coord;
  PETScWrappers::MPI::Vector completely_distributed_y_old_coord;

  unsigned int MESH_VELOCITY;
  PETScWrappers::MPI::Vector locally_relevant_mesh_velocity_u_Q2;
  PETScWrappers::MPI::Vector locally_relevant_mesh_velocity_v_Q2;
  PETScWrappers::MPI::Vector locally_relevant_body_velocity_u_Q2;
  PETScWrappers::MPI::Vector locally_relevant_body_velocity_v_Q2;
  PETScWrappers::MPI::Vector locally_relevant_mesh_velocity_u_Q1;
  PETScWrappers::MPI::Vector locally_relevant_mesh_velocity_v_Q1;
  PETScWrappers::MPI::Vector locally_relevant_body_velocity_u_Q1;
  PETScWrappers::MPI::Vector locally_relevant_body_velocity_v_Q1;
  PETScWrappers::MPI::Vector completely_distributed_mesh_velocity_u_Q1;
  PETScWrappers::MPI::Vector completely_distributed_mesh_velocity_v_Q1;
  PETScWrappers::MPI::Vector completely_distributed_mesh_velocity_u_Q2;
  PETScWrappers::MPI::Vector completely_distributed_mesh_velocity_v_Q2;
  PETScWrappers::MPI::Vector completely_distributed_aux_vector_in_Q2;

  PETScWrappers::MPI::SparseMatrix dummy_matrix_LS;
  PETScWrappers::MPI::SparseMatrix smoothing_matrix_u, smoothing_matrix_v;
  PETScWrappers::MPI::Vector smoothing_rhs_u;
  PETScWrappers::MPI::Vector smoothing_rhs_v;
  PETScWrappers::MPI::Vector smoothing_solver_solution;  
  std_cxx1x::shared_ptr<PETScWrappers::PreconditionBoomerAMG> preconditioner_smoothing_u, preconditioner_smoothing_v;
  std::map<typename DoFHandler<dim>::active_cell_iterator,double> min_h_on_cell;
  bool rebuild_smoothing_preconditioner;

  // SOLUTION VECTORS
  PETScWrappers::MPI::Vector locally_relevant_solution_phi;
  PETScWrappers::MPI::Vector locally_relevant_solution_u;
  PETScWrappers::MPI::Vector locally_relevant_solution_v;
  PETScWrappers::MPI::Vector locally_relevant_solution_p;
  PETScWrappers::MPI::Vector completely_distributed_solution_phi;
  PETScWrappers::MPI::Vector completely_distributed_solution_u;
  PETScWrappers::MPI::Vector completely_distributed_solution_v;
  PETScWrappers::MPI::Vector completely_distributed_solution_p;
  // BOUNDARY VECTORS
  std::vector<unsigned int> boundary_values_id_u;
  std::vector<unsigned int> boundary_values_id_v;
  std::vector<unsigned int> boundary_values_id_phi;
  std::vector<double> boundary_values_u;
  std::vector<double> boundary_values_v;
  std::vector<double> boundary_values_phi;

  double time;
  double time_step;
  double MIN_time_step;
  double MAX_time_step;
  double final_time;
  unsigned int timestep_number;
  double cfl;
  double umax;
  double min_h;

  double sharpness; 
  int sharpness_integer;

  unsigned int n_refinement;
  unsigned int output_number;
  double output_time;
  bool get_output;

  bool verbose;

  //FOR NAVIER STOKES
  double rho_fluid;
  double mu_fluid;
  double rho_air;
  double mu_air;
  double nu;
  double eps;
  double box_width;
  double box_mass;

  //FOR TRANSPORT
  double cK; //compression coeff
  double cE; //entropy-visc coeff
  unsigned int TRANSPORT_TIME_INTEGRATION;
  std::string ALGORITHM;
  unsigned int PROBLEM;
};

template <int dim>
MultiPhase<dim>::MultiPhase (const unsigned int degree_LS, 
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
  dof_handler_P (triangulation),
  fe_P (degree_U-1), 
  dof_handler_U_mesh(triangulation),
  fe_U_mesh(1),
  dof_handler_U_disp_field(triangulation),
  fe_U_disp_field(FE_Q<dim>(1),dim),
  pcout (std::cout,(Utilities::MPI::this_mpi_process(mpi_communicator)== 0)),
  rebuild_smoothing_preconditioner(true)
{}

template <int dim>
MultiPhase<dim>::~MultiPhase ()
{
  dof_handler_LS.clear ();
  dof_handler_U.clear ();
  dof_handler_P.clear ();
}

/////////////////////////////////////////
///////////////// SETUP /////////////////
/////////////////////////////////////////
template <int dim>
void MultiPhase<dim>::setup()
{ 
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
  // setup system P //
  dof_handler_P.distribute_dofs (fe_P);
  locally_owned_dofs_P = dof_handler_P.locally_owned_dofs ();
  DoFTools::extract_locally_relevant_dofs (dof_handler_P,
					   locally_relevant_dofs_P);
  // setup system U for disp field
  dof_handler_U_disp_field.distribute_dofs (fe_U_disp_field);
  locally_owned_dofs_U_disp_field = dof_handler_U_disp_field.locally_owned_dofs ();
  DoFTools::extract_locally_relevant_dofs (dof_handler_U_disp_field,
					   locally_relevant_dofs_U_disp_field);
  // setup system U mesh
  dof_handler_U_mesh.distribute_dofs (fe_U_mesh);
  locally_owned_dofs_U_mesh = dof_handler_U_mesh.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs (dof_handler_U_mesh,
					   locally_relevant_dofs_U_mesh);
  // INIT CONSTRAINTS
  init_constraints();
  // MOVING MESH
  // old and new coordinates
  x_old_coord.reinit(dof_handler_U_mesh.n_dofs());
  y_old_coord.reinit(dof_handler_U_mesh.n_dofs());
  x_new_coord.reinit(dof_handler_U_mesh.n_dofs());
  y_new_coord.reinit(dof_handler_U_mesh.n_dofs());
  // smoothing vectors
  locally_relevant_smoothed_lm1_x_new_coord.reinit(locally_owned_dofs_U_mesh,locally_relevant_dofs_U_mesh,mpi_communicator); 
  locally_relevant_smoothed_lm1_y_new_coord.reinit(locally_owned_dofs_U_mesh,locally_relevant_dofs_U_mesh,mpi_communicator); 
  completely_distributed_smoothed_x_new_coord.reinit(locally_owned_dofs_U_mesh,mpi_communicator); 
  completely_distributed_smoothed_y_new_coord.reinit(locally_owned_dofs_U_mesh,mpi_communicator); 
  completely_distributed_non_smooth_x_new_coord.reinit(locally_owned_dofs_U_mesh,mpi_communicator); 
  completely_distributed_non_smooth_y_new_coord.reinit(locally_owned_dofs_U_mesh,mpi_communicator); 
  completely_distributed_x_old_coord.reinit(locally_owned_dofs_U_mesh,mpi_communicator); 
  completely_distributed_y_old_coord.reinit(locally_owned_dofs_U_mesh,mpi_communicator); 
  ///////
  locally_relevant_mesh_velocity_u_Q2.reinit(locally_owned_dofs_U,locally_relevant_dofs_U,mpi_communicator); 
  locally_relevant_mesh_velocity_v_Q2.reinit(locally_owned_dofs_U,locally_relevant_dofs_U,mpi_communicator); 
  locally_relevant_body_velocity_u_Q2.reinit(locally_owned_dofs_U,locally_relevant_dofs_U,mpi_communicator); 
  locally_relevant_body_velocity_v_Q2.reinit(locally_owned_dofs_U,locally_relevant_dofs_U,mpi_communicator); 
  locally_relevant_mesh_velocity_u_Q2=0;
  locally_relevant_mesh_velocity_v_Q2=0;
  locally_relevant_body_velocity_u_Q2=0;
  locally_relevant_body_velocity_v_Q2=0;
  locally_relevant_mesh_velocity_u_Q1.reinit(locally_owned_dofs_U_mesh,locally_relevant_dofs_U_mesh,mpi_communicator); 
  locally_relevant_mesh_velocity_v_Q1.reinit(locally_owned_dofs_U_mesh,locally_relevant_dofs_U_mesh,mpi_communicator); 
  locally_relevant_body_velocity_u_Q1.reinit(locally_owned_dofs_U_mesh,locally_relevant_dofs_U_mesh,mpi_communicator); 
  locally_relevant_body_velocity_v_Q1.reinit(locally_owned_dofs_U_mesh,locally_relevant_dofs_U_mesh,mpi_communicator); 
  locally_relevant_mesh_velocity_u_Q1=0;
  locally_relevant_mesh_velocity_v_Q1=0;
  locally_relevant_body_velocity_u_Q1=0;
  locally_relevant_body_velocity_v_Q1=0;
  completely_distributed_mesh_velocity_u_Q1.reinit(locally_owned_dofs_U_mesh,mpi_communicator); 
  completely_distributed_mesh_velocity_v_Q1.reinit(locally_owned_dofs_U_mesh,mpi_communicator); 
  completely_distributed_mesh_velocity_u_Q2.reinit(locally_owned_dofs_U,mpi_communicator); 
  completely_distributed_mesh_velocity_v_Q2.reinit(locally_owned_dofs_U,mpi_communicator); 
  completely_distributed_aux_vector_in_Q2.reinit(locally_owned_dofs_U,mpi_communicator); 
  // init vectors for phi
  locally_relevant_solution_phi.reinit(locally_owned_dofs_LS,locally_relevant_dofs_LS,mpi_communicator);
  locally_relevant_solution_phi = 0;
  completely_distributed_solution_phi.reinit (locally_owned_dofs_P,mpi_communicator);
  //init vectors for u
  locally_relevant_solution_u.reinit (locally_owned_dofs_U,locally_relevant_dofs_U,mpi_communicator);
  locally_relevant_solution_u = 0;
  completely_distributed_solution_u.reinit (locally_owned_dofs_U,mpi_communicator);
  //init vectors for v                                           
  locally_relevant_solution_v.reinit (locally_owned_dofs_U,locally_relevant_dofs_U,mpi_communicator);
  locally_relevant_solution_v = 0;
  completely_distributed_solution_v.reinit (locally_owned_dofs_U,mpi_communicator);
  //init vectors for p
  locally_relevant_solution_p.reinit (locally_owned_dofs_P,locally_relevant_dofs_P,mpi_communicator);
  locally_relevant_solution_p = 0;
  completely_distributed_solution_p.reinit (locally_owned_dofs_P,mpi_communicator);
  // SMOOTHING MATRIX //
  DynamicSparsityPattern dsp (locally_relevant_dofs_U);
  DoFTools::make_sparsity_pattern (dof_handler_U,dsp,constraints_Q2,false);
  SparsityTools::distribute_sparsity_pattern (dsp,
					      dof_handler_U.n_locally_owned_dofs_per_processor(),
					      mpi_communicator,
					      locally_relevant_dofs_U);
  smoothing_matrix_u.reinit (mpi_communicator,
			     dsp,
			     dof_handler_U.n_locally_owned_dofs_per_processor(),
			     dof_handler_U.n_locally_owned_dofs_per_processor(),
			     Utilities::MPI::this_mpi_process(mpi_communicator));
  smoothing_matrix_v.reinit (mpi_communicator,
			     dsp,
			     dof_handler_U.n_locally_owned_dofs_per_processor(),
			     dof_handler_U.n_locally_owned_dofs_per_processor(),
			     Utilities::MPI::this_mpi_process(mpi_communicator));
  smoothing_matrix_u=0;
  smoothing_matrix_v=0;
  smoothing_rhs_u.reinit (locally_owned_dofs_U,mpi_communicator);
  smoothing_rhs_v.reinit (locally_owned_dofs_U,mpi_communicator);
  smoothing_solver_solution.reinit (locally_owned_dofs_U,mpi_communicator);
  // DUMMY MATRIX IN LS (to get sparsity pattern in LS)
  DynamicSparsityPattern dsp_LS (locally_relevant_dofs_LS);
  DoFTools::make_sparsity_pattern (dof_handler_LS,dsp_LS,constraints_Q1,false);
  SparsityTools::distribute_sparsity_pattern (dsp_LS,
					      dof_handler_LS.n_locally_owned_dofs_per_processor(),
					      mpi_communicator,
					      locally_relevant_dofs_LS);
  dummy_matrix_LS.reinit (mpi_communicator,
			  dsp_LS,
			  dof_handler_LS.n_locally_owned_dofs_per_processor(),
			  dof_handler_LS.n_locally_owned_dofs_per_processor(),
			  Utilities::MPI::this_mpi_process(mpi_communicator));
  dummy_matrix_LS=0;
  get_sparsity_pattern();
  get_boundary_map();
  get_maps_at_box();
  get_global_Q1_to_Q2_map();
}

template <int dim>
void MultiPhase<dim>::initial_condition(MappingQ<dim> &mapping)
{
  time=0;
  // Initial conditions //
  // init condition for phi
  completely_distributed_solution_phi = 0;
  VectorTools::interpolate(mapping,dof_handler_LS,
			   InitialPhi<dim>(PROBLEM, sharpness),
			   completely_distributed_solution_phi);
  constraints_Q1.distribute (completely_distributed_solution_phi);
  locally_relevant_solution_phi = completely_distributed_solution_phi;
  // init condition for u=0
  completely_distributed_solution_u = 0;
  VectorTools::interpolate(mapping,dof_handler_U,
			   ZeroFunction<dim>(),
			   completely_distributed_solution_u);
  constraints_Q2.distribute (completely_distributed_solution_u);
  locally_relevant_solution_u = completely_distributed_solution_u;
  // init condition for v
  completely_distributed_solution_v = 0;
  VectorTools::interpolate(mapping,dof_handler_U,
			   ZeroFunction<dim>(),
			   completely_distributed_solution_v);
  constraints_Q2.distribute (completely_distributed_solution_v);
  locally_relevant_solution_v = completely_distributed_solution_v;
  // init condition for p
  completely_distributed_solution_p = 0;
  VectorTools::interpolate(mapping,dof_handler_P,
			   ZeroFunction<dim>(),
			   completely_distributed_solution_p);
  constraints_Q1.distribute (completely_distributed_solution_p);
  locally_relevant_solution_p = completely_distributed_solution_p;
}
  
template <int dim>
void MultiPhase<dim>::init_constraints()
{
  constraints_Q1.clear ();
  constraints_Q1.reinit (locally_relevant_dofs_LS);
  DoFTools::make_hanging_node_constraints (dof_handler_LS, constraints_Q1);
  constraints_Q1.close ();
  constraints_Q2.clear ();
  constraints_Q2.reinit (locally_relevant_dofs_U);
  DoFTools::make_hanging_node_constraints (dof_handler_U, constraints_Q2);
  constraints_Q2.close ();
  // MOVING MESH
  constraints_disp_field.clear ();
  constraints_disp_field.reinit (locally_relevant_dofs_LS);
  DoFTools::make_hanging_node_constraints (dof_handler_LS, constraints_disp_field);
  constraints_disp_field.close ();
}

template <int dim>
void MultiPhase<dim>::get_boundary_values_U(MappingQ<dim> &mapping, double time)
{
  std::map<unsigned int, double> map_boundary_values_u;
  std::map<unsigned int, double> map_boundary_values_v;
  if (PROBLEM==SMALL_WAVE_PERTURBATION)
    { // no slip in bottom and top and slip in left and right
      //LEFT
      VectorTools::interpolate_boundary_values(mapping,dof_handler_U,0,ZeroFunction<dim>(),map_boundary_values_u); 
      // RIGHT
      VectorTools::interpolate_boundary_values(mapping,dof_handler_U,1,ZeroFunction<dim>(),map_boundary_values_u); 
      // BOTTOM 
      VectorTools::interpolate_boundary_values(mapping,dof_handler_U,2,ZeroFunction<dim>(),map_boundary_values_u); 
      VectorTools::interpolate_boundary_values(mapping,dof_handler_U,2,ZeroFunction<dim>(),map_boundary_values_v); 
      // TOP
      VectorTools::interpolate_boundary_values(mapping,dof_handler_U,3,ZeroFunction<dim>(),map_boundary_values_u); 
      VectorTools::interpolate_boundary_values(mapping,dof_handler_U,3,ZeroFunction<dim>(),map_boundary_values_v); 
    }
  else if (PROBLEM==BOX_PROBLEM)
    {
      //LEFT and Right: id=0,1
      VectorTools::interpolate_boundary_values(mapping,dof_handler_U,0,ZeroFunction<dim>(),map_boundary_values_u); 
      VectorTools::interpolate_boundary_values(mapping,dof_handler_U,1,ZeroFunction<dim>(),map_boundary_values_u); 
      // BOTTOM and TOP: id=2,3
      VectorTools::interpolate_boundary_values(mapping,dof_handler_U,2,ZeroFunction<dim>(),map_boundary_values_u); 
      VectorTools::interpolate_boundary_values(mapping,dof_handler_U,2,ZeroFunction<dim>(),map_boundary_values_v); 
      VectorTools::interpolate_boundary_values(mapping,dof_handler_U,3,ZeroFunction<dim>(),map_boundary_values_u); 
      VectorTools::interpolate_boundary_values(mapping,dof_handler_U,3,ZeroFunction<dim>(),map_boundary_values_v); 
      // BOX 
      VectorTools::interpolate_boundary_values(mapping,dof_handler_U,11,ZeroFunction<dim>(),map_boundary_values_u); 
      VectorTools::interpolate_boundary_values(mapping,dof_handler_U,11,MeshVelocityV<dim>(BOX_VELOCITY,time),map_boundary_values_v); 
      VectorTools::interpolate_boundary_values(mapping,dof_handler_U,12,ZeroFunction<dim>(),map_boundary_values_u); 
      VectorTools::interpolate_boundary_values(mapping,dof_handler_U,12,MeshVelocityV<dim>(BOX_VELOCITY,time),map_boundary_values_v); 
      VectorTools::interpolate_boundary_values(mapping,dof_handler_U,13,ZeroFunction<dim>(),map_boundary_values_u); 
      VectorTools::interpolate_boundary_values(mapping,dof_handler_U,13,MeshVelocityV<dim>(BOX_VELOCITY,time),map_boundary_values_v);
      VectorTools::interpolate_boundary_values(mapping,dof_handler_U,14,ZeroFunction<dim>(),map_boundary_values_u);
      VectorTools::interpolate_boundary_values(mapping,dof_handler_U,14,MeshVelocityV<dim>(BOX_VELOCITY,time),map_boundary_values_v);
    }
  else if (PROBLEM==FLOATING)
    {
      //LEFT and Right: id=0,1
      VectorTools::interpolate_boundary_values(mapping,dof_handler_U,0,ZeroFunction<dim>(),map_boundary_values_u); 
      VectorTools::interpolate_boundary_values(mapping,dof_handler_U,1,ZeroFunction<dim>(),map_boundary_values_u); 
      // BOTTOM and TOP: id=2,3
      VectorTools::interpolate_boundary_values(mapping,dof_handler_U,2,ZeroFunction<dim>(),map_boundary_values_u); 
      VectorTools::interpolate_boundary_values(mapping,dof_handler_U,2,ZeroFunction<dim>(),map_boundary_values_v); 
      VectorTools::interpolate_boundary_values(mapping,dof_handler_U,3,ZeroFunction<dim>(),map_boundary_values_u); 
      VectorTools::interpolate_boundary_values(mapping,dof_handler_U,3,ZeroFunction<dim>(),map_boundary_values_v); 
      // BOX
      IndexSet::ElementIterator idofs_iter = locally_owned_dofs_U_mesh.begin();  
      for (;idofs_iter!=locally_owned_dofs_U_mesh.end(); idofs_iter++)
	{
	  unsigned int gi = *idofs_iter;
	  if (is_dof_at_box[gi])
	    {
	      map_boundary_values_u[global_Q1_to_Q2_map[gi]] = locally_relevant_mesh_velocity_u_Q1(gi);
	      map_boundary_values_v[global_Q1_to_Q2_map[gi]] = locally_relevant_mesh_velocity_v_Q1(gi);
	    }
	}
    }
  else
    {
      pcout << "Error in type of PROBLEM at Boundary Conditions" << std::endl;
      abort();
    }
  boundary_values_id_u.resize(map_boundary_values_u.size());
  boundary_values_id_v.resize(map_boundary_values_v.size());
  boundary_values_u.resize(map_boundary_values_u.size());
  boundary_values_v.resize(map_boundary_values_v.size());
  std::map<unsigned int,double>::const_iterator boundary_value_u =map_boundary_values_u.begin();
  std::map<unsigned int,double>::const_iterator boundary_value_v =map_boundary_values_v.begin();
  
  for (int i=0; boundary_value_u !=map_boundary_values_u.end(); ++boundary_value_u, ++i)
    {
      boundary_values_id_u[i]=boundary_value_u->first;
      boundary_values_u[i]=boundary_value_u->second;
    }
  for (int i=0; boundary_value_v !=map_boundary_values_v.end(); ++boundary_value_v, ++i)
    {
      boundary_values_id_v[i]=boundary_value_v->first;
      boundary_values_v[i]=boundary_value_v->second;
    }
}


template <int dim>
void MultiPhase<dim>::set_boundary_inlet(MappingQ<dim> &mapping)
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
	      cell_U->face(face)->set_boundary_id(10); // SET ID 10 to inlet BOUNDARY (10 is an arbitrary number)
	  }    
}

template <int dim>
void MultiPhase<dim>::get_boundary_values_phi(MappingQ<dim> &mapping, 
					      std::vector<unsigned int> &boundary_values_id_phi,
					      std::vector<double> &boundary_values_phi)
{
  std::map<unsigned int, double> map_boundary_values_phi;
  unsigned int boundary_id=0;
  
  set_boundary_inlet(mapping);
  boundary_id=10; // inlet
  VectorTools::interpolate_boundary_values (mapping,dof_handler_LS,
					    boundary_id,BoundaryPhi<dim>(),map_boundary_values_phi);
  boundary_values_id_phi.resize(map_boundary_values_phi.size());
  boundary_values_phi.resize(map_boundary_values_phi.size());  
  std::map<unsigned int,double>::const_iterator boundary_value_phi = map_boundary_values_phi.begin();
  for (int i=0; boundary_value_phi !=map_boundary_values_phi.end(); ++boundary_value_phi, ++i)
    {
      boundary_values_id_phi[i]=boundary_value_phi->first;
      boundary_values_phi[i]=boundary_value_phi->second;
    }
}

template<int dim>
void MultiPhase<dim>::output_results(MappingQ<dim> &mapping)
{
  output_vectors(mapping);
  output_rho(mapping);
  output_number++;
}

template <int dim>
void MultiPhase<dim>::output_vectors(MappingQ<dim> &mapping)
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler (dof_handler_LS);  
  data_out.add_data_vector (locally_relevant_solution_phi, "phi");
  //data_out.attach_dof_handler (dof_handler_U);  
  //data_out.add_data_vector (locally_relevant_solution_u, "u");
  //data_out.add_data_vector (locally_relevant_solution_v, "v");
  data_out.build_patches (mapping);
  
  const std::string filename = ("sol_vectors-" +
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
	filenames.push_back ("sol_vectors-" +
			     Utilities::int_to_string (output_number, 3) +
			     "." +
			     Utilities::int_to_string (i, 4) +
			     ".vtu");
      
      std::ofstream master_output ((filename + ".pvtu").c_str());
      data_out.write_pvtu_record (master_output, filenames);
    }
}

template <int dim>
void MultiPhase<dim>::output_rho(MappingQ<dim> &mapping)
{
  Postprocessor<dim> postprocessor(eps,rho_air,rho_fluid);  
  DataOut<dim> data_out;
  data_out.attach_dof_handler (dof_handler_LS);  
  data_out.add_data_vector (locally_relevant_solution_phi, postprocessor);
  
  data_out.build_patches (mapping);
  
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
void MultiPhase<dim>::interpolate_from_Q1_to_Q2(PETScWrappers::MPI::Vector &vector_in_Q2,
						PETScWrappers::MPI::Vector &vector_in_Q1)
{
  const QTrapez<1>  q_trapez;
  const QIterated<dim> quad_points(q_trapez,degree_U);
  FEValues<dim> fe_values_U_mesh (fe_U_mesh, quad_points,
				  update_values    |  update_gradients |
				  update_quadrature_points |
				  update_JxW_values);
  const unsigned int n_q_points = quad_points.size(); //=num DOFs per Q2 cell

  std::vector<unsigned int> local_dof_indices_U_mesh (fe_U_mesh.dofs_per_cell);
  std::vector<unsigned int> local_dof_indices_U (fe_U.dofs_per_cell);

  std::vector<double>  values_with_quadrature_numbering (n_q_points);
  Vector<double>  values_with_dof_numbering (n_q_points);

  // loop around cells
  typename DoFHandler<dim>::active_cell_iterator
    cell_U_mesh = dof_handler_U_mesh.begin_active(),
    endc_U_mesh = dof_handler_U_mesh.end();
  typename DoFHandler<dim>::active_cell_iterator
    cell_U = dof_handler_U.begin_active();
  
  for (; cell_U_mesh!=endc_U_mesh; ++cell_U_mesh, ++cell_U)
    if (cell_U_mesh->is_locally_owned())
      {
	fe_values_U_mesh.reinit(cell_U_mesh);
	cell_U_mesh->get_dof_indices (local_dof_indices_U_mesh);
	cell_U->get_dof_indices (local_dof_indices_U);
	fe_values_U_mesh.get_function_values(vector_in_Q1,values_with_quadrature_numbering);
	// nodal values are aranged based on quadrature points
	values_with_dof_numbering[0] = values_with_quadrature_numbering[0];
	values_with_dof_numbering[1] = values_with_quadrature_numbering[2];
	values_with_dof_numbering[2] = values_with_quadrature_numbering[6];
	values_with_dof_numbering[3] = values_with_quadrature_numbering[8];
	values_with_dof_numbering[4] = values_with_quadrature_numbering[3];
	values_with_dof_numbering[5] = values_with_quadrature_numbering[5];
	values_with_dof_numbering[6] = values_with_quadrature_numbering[1];
	values_with_dof_numbering[7] = values_with_quadrature_numbering[7];
	values_with_dof_numbering[8] = values_with_quadrature_numbering[4];
	
	for (unsigned int i=0; i<n_q_points; i++)
	  completely_distributed_aux_vector_in_Q2[local_dof_indices_U[i]]=values_with_dof_numbering[i];
    }
  completely_distributed_aux_vector_in_Q2.compress(VectorOperation::insert);
  vector_in_Q2 = completely_distributed_aux_vector_in_Q2;
}

template<int dim>
void MultiPhase<dim>::get_sparsity_pattern()
{
  sparsity_pattern.clear();
  // loop on DOFs
  IndexSet::ElementIterator idofs_iter = locally_owned_dofs_LS.begin();
  int ncolumns;
  const int *gj; 
  const double *Mi;

  for (;idofs_iter!=locally_owned_dofs_LS.end(); idofs_iter++)
    {
      int gi = *idofs_iter;      
      // get i-th row of mass matrix (dummy, I just need the indices gj)
      MatGetRow(dummy_matrix_LS,gi,&ncolumns,&gj,&Mi);
      sparsity_pattern[gi] = std::vector<types::global_dof_index>(gj,gj+ncolumns);
      MatRestoreRow(dummy_matrix_LS,gi,&ncolumns,&gj,&Mi);
    }
}

template<int dim>
void MultiPhase<dim>::get_boundary_map()
{
  is_dof_at_boundary.clear();
  unsigned int dofs_per_face = fe_LS.dofs_per_face;
  std::vector<unsigned int> local_dof_indices (dofs_per_face);

  IndexSet::ElementIterator idofs_iter = locally_owned_dofs_LS.begin();  
  for (;idofs_iter!=locally_owned_dofs_LS.end(); idofs_iter++)
    is_dof_at_boundary[*idofs_iter] = false;

  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler_LS.begin_active(),
    endc = dof_handler_LS.end();

  for (; cell!=endc; ++cell)
    if (cell->is_locally_owned() && cell->at_boundary())
      for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
        if (cell->face(face)->at_boundary())
          {	    
	    cell->face(face)->get_dof_indices (local_dof_indices) ;
	    for (unsigned int i=0; i<dofs_per_face; ++i)
	      is_dof_at_boundary[local_dof_indices[i]] = true; 
	  }
}

template<int dim>
void MultiPhase<dim>::get_global_Q1_to_Q2_map()
{
  global_Q1_to_Q2_map.clear();

  std::vector<unsigned int> local_dof_indices_Q1 (fe_U_mesh.dofs_per_cell);
  std::vector<unsigned int> local_dof_indices_Q2 (fe_U.dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator
    cell_Q1 = dof_handler_U_mesh.begin_active(),
    endc_Q1 = dof_handler_U_mesh.end(),
    cell_Q2 = dof_handler_U.begin_active();

  for (; cell_Q1!=endc_Q1; ++cell_Q1, ++cell_Q2)
    if (cell_Q1->is_locally_owned())
      {
	cell_Q1->get_dof_indices(local_dof_indices_Q1);
	cell_Q2->get_dof_indices(local_dof_indices_Q2);
	for (unsigned int i=0; i<fe_U_mesh.dofs_per_cell; ++i)
	  global_Q1_to_Q2_map[local_dof_indices_Q1[i]] = local_dof_indices_Q2[i];
      }
}

template<int dim>
void MultiPhase<dim>::get_maps_at_box()
{
  cell_U_mesh_at_box.clear();
  is_dof_at_box.clear();

  unsigned int dofs_per_face = fe_LS.dofs_per_face;
  std::vector<unsigned int> local_dof_indices (dofs_per_face);

  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler_U_mesh.begin_active(),
    endc = dof_handler_U_mesh.end();

  IndexSet::ElementIterator idofs_iter = locally_owned_dofs_U_mesh.begin();  
  for (;idofs_iter!=locally_owned_dofs_U_mesh.end(); idofs_iter++)
    is_dof_at_box[*idofs_iter] = false;

  for (; cell!=endc; ++cell)
    if (cell->is_locally_owned())
      cell_U_mesh_at_box[cell]=false;

  for (cell=dof_handler_U_mesh.begin_active(); cell!=endc; ++cell)
    if (cell->is_locally_owned() && cell->at_boundary())
      for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
	if (cell->face(face)->at_boundary())
	  {
	    unsigned int id = (unsigned int) cell->face(face)->boundary_id();
	    if (id==11 || id==12 || id==13 || id==14)
	      {
		cell_U_mesh_at_box[cell] = true;
		cell->face(face)->get_dof_indices(local_dof_indices);
		for (unsigned int i=0; i<dofs_per_face; ++i)
		  is_dof_at_box[local_dof_indices[i]] = true; 
	      }
	  }
}

template <int dim>
int MultiPhase<dim>::solve(const ConstraintMatrix &constraints, 
			    PETScWrappers::MPI::SparseMatrix &Matrix,
			    std_cxx1x::shared_ptr<PETScWrappers::PreconditionBoomerAMG> preconditioner,
			    PETScWrappers::MPI::Vector &completely_distributed_solution,
			    const PETScWrappers::MPI::Vector &rhs)
{
  // all vectors are NON-GHOSTED
  SolverControl solver_control (dof_handler_LS.n_dofs(), 1E-6);
  PETScWrappers::SolverCG solver(solver_control, mpi_communicator);
  constraints.distribute (completely_distributed_solution);
  solver.solve (Matrix, completely_distributed_solution, rhs, *preconditioner);
  constraints.distribute (completely_distributed_solution);
  if (verbose==true) pcout << "   Solved Smoothing in " << solver_control.last_step() << " iterations." << std::endl;
  return solver_control.last_step();
}

template<int dim>
void MultiPhase<dim>::get_smoothed_velocity_via_gravity_center(double dt, Vector<double> &old_coord,
							       PETScWrappers::MPI::Vector &velocity_to_smooth_u,
							       PETScWrappers::MPI::Vector &velocity_to_smooth_v,
							       PETScWrappers::MPI::Vector &smoothed_velocity_u,
							       PETScWrappers::MPI::Vector &smoothed_velocity_v)
{
  int L = 8; // number of loops for smoothing
  double omega = 0.1;

  compute_new_coordinates_from_mesh_velocity_vectors(dt,
						     velocity_to_smooth_u,velocity_to_smooth_v, //
						     old_coord,
						     completely_distributed_x_old_coord, 
						     completely_distributed_y_old_coord,
						     completely_distributed_non_smooth_x_new_coord,
						     completely_distributed_non_smooth_y_new_coord);

  locally_relevant_smoothed_lm1_x_new_coord = completely_distributed_non_smooth_x_new_coord;
  locally_relevant_smoothed_lm1_y_new_coord = completely_distributed_non_smooth_y_new_coord;
  // loops for smoothing 
  for (int l=0; l<L; l++)
    {
      // loop on locally owned i-DOFs (rows)
      IndexSet::ElementIterator idofs_iter = locally_owned_dofs_LS.begin();
      for (;idofs_iter!=locally_owned_dofs_LS.end(); idofs_iter++)
	{
	  int gi = *idofs_iter;
	  int ncolumns = sparsity_pattern[gi].size();
	  std::vector<double> x_new_coord_subvector(ncolumns);
	  std::vector<double> y_new_coord_subvector(ncolumns);
	  
	  locally_relevant_smoothed_lm1_x_new_coord.extract_subvector_to(sparsity_pattern[gi],x_new_coord_subvector);
	  locally_relevant_smoothed_lm1_y_new_coord.extract_subvector_to(sparsity_pattern[gi],y_new_coord_subvector);

	  double xi = locally_relevant_smoothed_lm1_x_new_coord(gi);
	  double yi = locally_relevant_smoothed_lm1_y_new_coord(gi);
	  
	  if (is_dof_at_boundary[gi])
	    {
	      if (xi==0 || xi==1) //LEFT OR RIGHT BOUNDARIES
		{
		  completely_distributed_smoothed_x_new_coord(gi) = xi;
		  if (yi==0 || yi==1)
		    completely_distributed_smoothed_y_new_coord(gi) = yi;
		  else 
		    completely_distributed_smoothed_y_new_coord(gi) 
		      =(std::accumulate(y_new_coord_subvector.begin(),y_new_coord_subvector.end(),0.0)-yi)/(ncolumns-1.);
		}
	      else if (yi==0 || yi==1) // BOTTOM OR TOP BOUNDARY
		{
		  completely_distributed_smoothed_y_new_coord(gi) = yi;		  
		  if (xi==0 || xi==1)
		    completely_distributed_smoothed_x_new_coord(gi) =xi;
		  else 
		    completely_distributed_smoothed_x_new_coord(gi) 
		      =(std::accumulate(x_new_coord_subvector.begin(),x_new_coord_subvector.end(),0.0)-xi)/(ncolumns-1.);
		}
	      else //BOX
		{
		  completely_distributed_smoothed_x_new_coord(gi) = xi;
		  completely_distributed_smoothed_y_new_coord(gi) = yi;
		}
	    }	      
	  else
	    {
	      completely_distributed_smoothed_x_new_coord(gi) 
		=(std::accumulate(x_new_coord_subvector.begin(),x_new_coord_subvector.end(),0.0)-xi)/(ncolumns-1.);
	      completely_distributed_smoothed_y_new_coord(gi) 
		=(std::accumulate(y_new_coord_subvector.begin(),y_new_coord_subvector.end(),0.0)-yi)/(ncolumns-1.);
	    }
	}
      completely_distributed_smoothed_x_new_coord.compress(VectorOperation::insert);  
      completely_distributed_smoothed_y_new_coord.compress(VectorOperation::insert);  
      locally_relevant_smoothed_lm1_x_new_coord = completely_distributed_smoothed_x_new_coord;
      locally_relevant_smoothed_lm1_y_new_coord = completely_distributed_smoothed_y_new_coord;
    }
  // combine smoothed coord and non-smoothed coord
  completely_distributed_smoothed_x_new_coord.equ(1.0-omega,locally_relevant_smoothed_lm1_x_new_coord);
  completely_distributed_smoothed_y_new_coord.equ(1.0-omega,locally_relevant_smoothed_lm1_y_new_coord);
  completely_distributed_smoothed_x_new_coord.add(omega,completely_distributed_non_smooth_x_new_coord);
  completely_distributed_smoothed_y_new_coord.add(omega,completely_distributed_non_smooth_y_new_coord);
  
  // velocity = (smoothed_new_coord - old_coord)/dt
  IndexSet::ElementIterator idofs_iter = locally_owned_dofs_U_mesh.begin();  
  for (;idofs_iter!=locally_owned_dofs_U_mesh.end(); idofs_iter++)
    {
      int gi = *idofs_iter;
      completely_distributed_mesh_velocity_u_Q1(gi) = (completely_distributed_smoothed_x_new_coord(gi) 
						       - completely_distributed_x_old_coord(gi))/dt;
      completely_distributed_mesh_velocity_v_Q1(gi) = (completely_distributed_smoothed_y_new_coord(gi) 
						       - completely_distributed_y_old_coord(gi))/dt;
    }
  completely_distributed_mesh_velocity_u_Q1.compress(VectorOperation::insert);
  completely_distributed_mesh_velocity_v_Q1.compress(VectorOperation::insert);
  smoothed_velocity_u = completely_distributed_mesh_velocity_u_Q1;
  smoothed_velocity_v = completely_distributed_mesh_velocity_v_Q1;
}

template<int dim>
void MultiPhase<dim>::get_smoothed_velocity_via_laplacian(MappingQ<dim> &mapping,
							  PETScWrappers::MPI::Vector &velocity_to_smooth_u,
							  PETScWrappers::MPI::Vector &velocity_to_smooth_v,
							  PETScWrappers::MPI::Vector &smoothed_velocity_u,
							  PETScWrappers::MPI::Vector &smoothed_velocity_v)
{
  smoothing_matrix_u=0;
  smoothing_matrix_v=0;
  smoothing_rhs_u=0;
  smoothing_rhs_v=0;

  const QGauss<dim>  quadrature_formula(degree_U+1);
  FEValues<dim> fe_values_U (mapping, fe_U, quadrature_formula,
			     update_values    |  update_gradients | 
			     update_quadrature_points |
			     update_JxW_values);
  
  const unsigned int   dofs_per_cell = fe_U.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();

  std::vector<double>  u (n_q_points); 
  std::vector<double>  v (n_q_points); 
  
  Vector<double> cell_rhs_u(dofs_per_cell);
  Vector<double> cell_rhs_v(dofs_per_cell);
  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  std::vector<unsigned int> local_dof_indices (dofs_per_cell);

  std::vector<double> shape_values(dofs_per_cell);
  std::vector<Tensor<1, dim> > shape_grads(dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler_U.begin_active(),
    endc = dof_handler_U.end();
  
  for (; cell!=endc; ++cell)
    if (cell->is_locally_owned())
      {
	double h2 = std::pow(min_h_on_cell[cell],2);
	
	cell_matrix = 0;
	cell_rhs_u = 0;
	cell_rhs_v = 0;

	fe_values_U.reinit (cell);
	fe_values_U.get_function_values(velocity_to_smooth_u,u);
	fe_values_U.get_function_values(velocity_to_smooth_v,v);

	for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
	  {
	    const double JxW = fe_values_U.JxW(q_point);
	    for (unsigned int i=0; i<dofs_per_cell; ++i)
	      {
		shape_values[i] = fe_values_U.shape_value(i,q_point);
		shape_grads[i] = fe_values_U.shape_grad(i,q_point);
	      }
	    for (unsigned int i=0; i<dofs_per_cell; ++i)
	      {
		for (unsigned int j=0; j<dofs_per_cell; ++j)
		  cell_matrix(i,j) += (shape_values[i]*shape_values[j]
				       +0.1*h2*(shape_grads[i]*shape_grads[j])
				       )*JxW;
		cell_rhs_u(i) += u[q_point]*shape_values[i]*JxW; 
		cell_rhs_v(i) += v[q_point]*shape_values[i]*JxW; 
	      }
	  }
	// distribute
	cell->get_dof_indices (local_dof_indices);
	constraints_Q2.distribute_local_to_global (cell_matrix,local_dof_indices,smoothing_matrix_u);
	constraints_Q2.distribute_local_to_global (cell_rhs_u,local_dof_indices,smoothing_rhs_u);
	constraints_Q2.distribute_local_to_global (cell_rhs_v,local_dof_indices,smoothing_rhs_v);
      }
  // compress
  smoothing_matrix_u.compress(VectorOperation::add);
  smoothing_matrix_v.copy_from(smoothing_matrix_u);
  smoothing_rhs_u.compress(VectorOperation::add);
  smoothing_rhs_v.compress(VectorOperation::add);
  // BOUNDARIES 
  smoothing_rhs_u.set(boundary_values_id_u,boundary_values_u);
  smoothing_rhs_u.compress(VectorOperation::insert);
  smoothing_rhs_v.set(boundary_values_id_v,boundary_values_v);
  smoothing_rhs_v.compress(VectorOperation::insert);
  smoothing_matrix_u.clear_rows(boundary_values_id_u,1);
  smoothing_matrix_u.compress(VectorOperation::insert);
  smoothing_matrix_v.clear_rows(boundary_values_id_v,1);
  smoothing_matrix_v.compress(VectorOperation::insert);
  // PRECONDITIONER
  if (rebuild_smoothing_preconditioner)
    {
      preconditioner_smoothing_u.reset
	(new PETScWrappers::PreconditionBoomerAMG(smoothing_matrix_u,
						  PETScWrappers::PreconditionBoomerAMG::AdditionalData(true)));
      preconditioner_smoothing_v.reset
	(new PETScWrappers::PreconditionBoomerAMG(smoothing_matrix_v,
						  PETScWrappers::PreconditionBoomerAMG::AdditionalData(true)));
    }
  smoothing_solver_solution=velocity_to_smooth_u;
  int smoothing_iter_u=solve(constraints_Q2,smoothing_matrix_u,preconditioner_smoothing_u,
			     smoothing_solver_solution,smoothing_rhs_u);
  smoothed_velocity_u=smoothing_solver_solution;
  smoothing_solver_solution=velocity_to_smooth_v;
  int smoothing_iter_v=solve(constraints_Q2,smoothing_matrix_v,preconditioner_smoothing_v,
			     smoothing_solver_solution,smoothing_rhs_v);
  smoothed_velocity_v=smoothing_solver_solution;
  if (smoothing_iter_u > NUM_ITER_TO_RECOMPUTE_PRECONDITIONER || smoothing_iter_v > NUM_ITER_TO_RECOMPUTE_PRECONDITIONER)
    rebuild_smoothing_preconditioner=true;
  else
    rebuild_smoothing_preconditioner=false;
}

template<int dim>
double MultiPhase<dim>::get_time_step(MappingQ<dim> &mapping, Vector<double> coord)
{  
  min_h=1E10;
  Vector<double> umax_per_cell (triangulation.n_active_cells());
  VectorTools::integrate_difference (mapping,dof_handler_U,
				     locally_relevant_solution_u,
				     ZeroFunction<dim>(),
				     umax_per_cell,
				     QGauss<dim>(degree_U+1),
				     VectorTools::Linfty_norm);
  Vector<double> vmax_per_cell (triangulation.n_active_cells());
  VectorTools::integrate_difference (mapping,dof_handler_U,
				     locally_relevant_solution_v,
				     ZeroFunction<dim>(),
				     vmax_per_cell,
				     QGauss<dim>(degree_U+1),
				     VectorTools::Linfty_norm);
  // get dt
  const unsigned int   dofs_per_cell = fe_U_disp_field.dofs_per_cell;
  std::vector<unsigned int> local_dof_indices (dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler_U_disp_field.begin_active(),
    endc = dof_handler_U_disp_field.end();
  typename DoFHandler<dim>::active_cell_iterator
    cell_U_uncoupled = dof_handler_U.begin_active();

  double xi, yi, xj, yj, dist, h;
  int k=0;
  double dt=1000;

  double UMAX = 0;
  for (; cell!=endc; ++cell, ++k, ++cell_U_uncoupled)
    if (cell->is_locally_owned())
      {
  	cell->get_dof_indices (local_dof_indices);
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
	min_h_on_cell[cell_U_uncoupled]=h;
	min_h=std::min(min_h,h);
	UMAX = std::max(UMAX,std::sqrt(std::pow(umax_per_cell(k),2)+std::pow(vmax_per_cell(k),2)));	
	UMAX = std::max(UMAX, 1.0);
	dt = cfl*h/UMAX;
	dt = std::min(MAX_time_step,std::max(dt,MIN_time_step));
      }
  std::cout << "****** UMAX: " << UMAX << std::endl;
  Utilities::MPI::min(min_h,mpi_communicator);
  return Utilities::MPI::min(dt,mpi_communicator);
}

template<int dim>
void MultiPhase<dim>::get_interpolated_mesh_velocity(MappingQ<dim> &mapping)
{
  completely_distributed_mesh_velocity_u_Q2 = 0;
  VectorTools::interpolate(mapping,dof_handler_U,
			   MeshVelocityU<dim>(MESH_VELOCITY,time),
			   completely_distributed_mesh_velocity_u_Q2);
  constraints_Q2.distribute (completely_distributed_mesh_velocity_u_Q2);
  locally_relevant_mesh_velocity_u_Q2=completely_distributed_mesh_velocity_u_Q2;
  completely_distributed_mesh_velocity_v_Q2 = 0;
  VectorTools::interpolate(mapping,dof_handler_U,
			   MeshVelocityV<dim>(MESH_VELOCITY,time),
			   completely_distributed_mesh_velocity_v_Q2);
  constraints_Q2.distribute (completely_distributed_mesh_velocity_v_Q2);  
  locally_relevant_mesh_velocity_v_Q2=completely_distributed_mesh_velocity_v_Q2;
}

template<int dim>
void MultiPhase<dim>::compute_vertex_coord(Vector<double> &old_coord)
{
  x_old_coord = 0;
  y_old_coord = 0;    
  old_coord = 0;
  const unsigned int   dofs_per_cell = fe_U_disp_field.dofs_per_cell;
  std::vector<unsigned int> local_dof_indices (dofs_per_cell);
  std::vector<unsigned int> local_dof_indices_U_mesh (fe_U_mesh.dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler_U_disp_field.begin_active(),
    endc = dof_handler_U_disp_field.end();

  for (; cell!=endc; ++cell)
    if (cell->is_locally_owned())
      {
  	cell->get_dof_indices (local_dof_indices);
	for (unsigned int i=0; i<GeometryInfo<dim>::vertices_per_cell; ++i)
	  {
	    old_coord(local_dof_indices[fe_U_disp_field.component_to_system_index(0,i)]) 
	      = cell->vertex(i)[0];
	    old_coord(local_dof_indices[fe_U_disp_field.component_to_system_index(1,i)]) 
	      = cell->vertex(i)[1];
	    x_old_coord(local_dof_indices_U_mesh[i]) = cell->vertex(i)[0];
	    y_old_coord(local_dof_indices_U_mesh[i]) = cell->vertex(i)[1];
	  }
      }
}

template<int dim>
void MultiPhase<dim>::compute_mesh_displacement_via_vectors_in_Q1(double dt, 
								  PETScWrappers::MPI::Vector &mesh_vel_u,
								  PETScWrappers::MPI::Vector &mesh_vel_v,
								  Vector<double> &old_coord,
								  Vector<double> &new_coord,
								  Vector<double> &mesh_disp)
{
  x_new_coord = 0;
  y_new_coord = 0;
  const unsigned int   dofs_per_cell = fe_U_disp_field.dofs_per_cell;
  std::vector<unsigned int> local_dof_indices (dofs_per_cell);
  std::vector<unsigned int> local_dof_indices_U_uncoupled (fe_U_mesh.dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler_U_disp_field.begin_active(),
    endc = dof_handler_U_disp_field.end();
  typename DoFHandler<dim>::active_cell_iterator cell_U_uncoupled = dof_handler_U_mesh.begin_active();

  double vx,vy,x,y;
  double step_dispX, step_dispY, new_coordX, new_coordY, DispX, DispY;
  for (; cell!=endc; ++cell, ++cell_U_uncoupled)
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
	    // compute displacement on a single time step
	    step_dispX = vx*dt;
	    step_dispY = vy*dt;
	    // compute new coord
	    new_coordX = x+step_dispX;
	    new_coordY = y+step_dispY;
	    new_coord(local_dof_indices[fe_U_disp_field.component_to_system_index(0,i)]) = new_coordX;
	    new_coord(local_dof_indices[fe_U_disp_field.component_to_system_index(1,i)]) = new_coordY;
	    x_new_coord(local_dof_indices_U_uncoupled[i]) = new_coordX;
	    y_new_coord(local_dof_indices_U_uncoupled[i]) = new_coordY;
	    // compute global displacement; i.e., from original coord
	    DispX = new_coordX - cell->vertex(i)[0]; //new_coord - original_coord
	    DispY = new_coordY - cell->vertex(i)[1];

	    mesh_disp(local_dof_indices[fe_U_disp_field.component_to_system_index(0,i)])=DispX;
	    mesh_disp(local_dof_indices[fe_U_disp_field.component_to_system_index(1,i)])=DispY;
	  }
      }
}

template<int dim>
void MultiPhase<dim>::compute_mesh_displacement_via_vectors_in_Q2(double dt, 
								  PETScWrappers::MPI::Vector &mesh_vel_u,
								  PETScWrappers::MPI::Vector &mesh_vel_v,
								  Vector<double> &old_coord,
								  Vector<double> &new_coord,
								  Vector<double> &mesh_disp)
{
  const unsigned int   dofs_per_cell = fe_U_disp_field.dofs_per_cell;
  std::vector<unsigned int> local_dof_indices (dofs_per_cell);
  std::vector<unsigned int> local_dof_indices_U_uncoupled (fe_U.dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler_U_disp_field.begin_active(),
    endc = dof_handler_U_disp_field.end();
  typename DoFHandler<dim>::active_cell_iterator cell_U_uncoupled = dof_handler_U.begin_active();

  double vx,vy,x,y;
  double step_dispX, step_dispY, new_coordX, new_coordY, DispX, DispY;
  for (; cell!=endc; ++cell, ++cell_U_uncoupled)
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
	    // compute displacement on a single time step
	    step_dispX = vx*dt;
	    step_dispY = vy*dt;
	    // compute new coord
	    new_coordX = x+step_dispX;
	    new_coordY = y+step_dispY;
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

template<int dim>
void MultiPhase<dim>::compute_new_coordinates_from_mesh_velocity_vectors(double dt, 
									 PETScWrappers::MPI::Vector &mesh_vel_u,
									 PETScWrappers::MPI::Vector &mesh_vel_v,
									 Vector<double> &old_coord,
									 PETScWrappers::MPI::Vector &x_old_coord,
									 PETScWrappers::MPI::Vector &y_old_coord,
									 PETScWrappers::MPI::Vector &x_new_coord,
									 PETScWrappers::MPI::Vector &y_new_coord)
{
  //NOTE: velocity is in Q1 or Q2. This works since first 4 DOFs in Q1 and Q2 have the same location
  const unsigned int   dofs_per_cell = fe_U_disp_field.dofs_per_cell;
  std::vector<unsigned int> local_dof_indices (dofs_per_cell);
  std::vector<unsigned int> local_dof_indices_U (fe_U.dofs_per_cell);
  std::vector<unsigned int> local_dof_indices_U_mesh (fe_U_mesh.dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler_U_disp_field.begin_active(),
    endc = dof_handler_U_disp_field.end();
  typename DoFHandler<dim>::active_cell_iterator cell_U = dof_handler_U.begin_active();
  typename DoFHandler<dim>::active_cell_iterator cell_U_mesh = dof_handler_U_mesh.begin_active();

  double vx,vy,x,y;
  for (; cell!=endc; ++cell, ++cell_U, ++cell_U_mesh)
    if (cell->is_locally_owned())
      {
  	cell->get_dof_indices (local_dof_indices);
  	cell_U->get_dof_indices (local_dof_indices_U);
  	cell_U_mesh->get_dof_indices (local_dof_indices_U_mesh);
	for (unsigned int i=0; i<GeometryInfo<dim>::vertices_per_cell; ++i)
	  {
	    x = old_coord(local_dof_indices[fe_U_disp_field.component_to_system_index(0,i)]);
	    y = old_coord(local_dof_indices[fe_U_disp_field.component_to_system_index(1,i)]);
	    vx = mesh_vel_u(local_dof_indices_U[i]);
	    vy = mesh_vel_v(local_dof_indices_U[i]);
	    // decompose old coord
	    x_old_coord(local_dof_indices_U_mesh[i]) = x;
	    y_old_coord(local_dof_indices_U_mesh[i]) = y;
	    // compute new coord
	    x_new_coord(local_dof_indices_U_mesh[i]) = x + vx*dt;
	    y_new_coord(local_dof_indices_U_mesh[i]) = y + vy*dt;
	  }
      }
  x_old_coord.compress(VectorOperation::insert);    
  y_old_coord.compress(VectorOperation::insert);    
  x_new_coord.compress(VectorOperation::insert);    
  y_new_coord.compress(VectorOperation::insert);    
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
/** boundary id is as follows:
 *   9-----3------3-----10
 *   |     |      |      |
 *   0-----144-14-160----1
 *   |     11     12     |
 *   0-----80--13-96-----1
 *   |     |      |      |
 *   5-----2------2------6
 */
template<int dim>
void MultiPhase<dim>::make_structured_box(parallel::distributed::Triangulation<dim> &tria)
{
           
  static const Point<dim> vertices_1[]
    = {Point<dim> (0,0), Point<dim> (0.35,0),Point<dim> (0.65,0),Point<dim> (1,0),
       Point<dim> (0,0.35), Point<dim> (0.35,0.35),Point<dim> (0.65,0.35),Point<dim> (1,0.35),
       Point<dim> (0,0.65), Point<dim> (0.35,0.65),Point<dim> (0.65,0.65),Point<dim> (1,0.65),
       Point<dim> (0,1), Point<dim> (0.35,1),Point<dim> (0.65,1),Point<dim> (1,1)};

  const unsigned int
    n_vertices = sizeof(vertices_1) / sizeof(vertices_1[0]);
  const std::vector<Point<dim> > vertices (&vertices_1[0],&vertices_1[n_vertices]);
  //**************************************************
  static const int cell_vertices[][GeometryInfo<dim>::vertices_per_cell]
    = {{0, 1, 4, 5},
       {1, 2, 5, 6},
       {2, 3, 6, 7},
       {4, 5, 8, 9},
       {6, 7,10,11},
       {8, 9,12,13},
       {9,10,13,14},
       {10,11,14,15}
  };
  const unsigned int
    n_cells = sizeof(cell_vertices) / sizeof(cell_vertices[0]);
  std::vector<CellData<dim> > cells (n_cells, CellData<dim>());
  for (unsigned int i=0; i<n_cells; ++i)
    {
      for (unsigned int j=0;j<GeometryInfo<dim>::vertices_per_cell;++j)
	cells[i].vertices[j] = cell_vertices[i][j];
      cells[i].material_id = 0;
    }
  //**************************************************
  SubCellData subcelldata;
  subcelldata.boundary_lines.resize(16);
  
  subcelldata.boundary_lines[0].vertices[0]=0;
  subcelldata.boundary_lines[0].vertices[1]=1;
  subcelldata.boundary_lines[0].material_id=2;
  
  subcelldata.boundary_lines[1].vertices[0]=1;
  subcelldata.boundary_lines[1].vertices[1]=2;
  subcelldata.boundary_lines[1].material_id=2;
  
  subcelldata.boundary_lines[2].vertices[0]=2;
  subcelldata.boundary_lines[2].vertices[1]=3;
  subcelldata.boundary_lines[2].material_id=2;
  
  subcelldata.boundary_lines[3].vertices[0]=3;
  subcelldata.boundary_lines[3].vertices[1]=7;
  subcelldata.boundary_lines[3].material_id=1;
  
  subcelldata.boundary_lines[4].vertices[0]=7;
  subcelldata.boundary_lines[4].vertices[1]=11;
  subcelldata.boundary_lines[4].material_id=1;
  
  subcelldata.boundary_lines[5].vertices[0]=11;
  subcelldata.boundary_lines[5].vertices[1]=15;
  subcelldata.boundary_lines[5].material_id=1;
  
  subcelldata.boundary_lines[6].vertices[0]=15;
  subcelldata.boundary_lines[6].vertices[1]=14;
  subcelldata.boundary_lines[6].material_id=3;
  
  subcelldata.boundary_lines[7].vertices[0]=14;
  subcelldata.boundary_lines[7].vertices[1]=13;
  subcelldata.boundary_lines[7].material_id=3;
  
  subcelldata.boundary_lines[8].vertices[0]=13;
  subcelldata.boundary_lines[8].vertices[1]=12;
  subcelldata.boundary_lines[8].material_id=3;
  
  subcelldata.boundary_lines[9].vertices[0]=12;
  subcelldata.boundary_lines[9].vertices[1]=8;
  subcelldata.boundary_lines[9].material_id=0;
  
  subcelldata.boundary_lines[10].vertices[0]=8;
  subcelldata.boundary_lines[10].vertices[1]=4;
  subcelldata.boundary_lines[10].material_id=0;
  
  subcelldata.boundary_lines[11].vertices[0]=4;
  subcelldata.boundary_lines[11].vertices[1]=0;
  subcelldata.boundary_lines[11].material_id=0;
  
  subcelldata.boundary_lines[12].vertices[0]=6;
  subcelldata.boundary_lines[12].vertices[1]=5;
  subcelldata.boundary_lines[12].material_id=13;
  
  subcelldata.boundary_lines[13].vertices[0]=10;
  subcelldata.boundary_lines[13].vertices[1]=6;
  subcelldata.boundary_lines[13].material_id=12;
  
  subcelldata.boundary_lines[14].vertices[0]=9;
  subcelldata.boundary_lines[14].vertices[1]=10;
  subcelldata.boundary_lines[14].material_id=14;
  
  subcelldata.boundary_lines[15].vertices[0]=5;
  subcelldata.boundary_lines[15].vertices[1]=9;
  subcelldata.boundary_lines[15].material_id=11;
  
  //**************************************************
  tria.create_triangulation (vertices,cells,subcelldata);
}

template<int dim>
void MultiPhase<dim>::get_mu(double &mu, double phi) {
  double H=0;
  // get rho, nu
  if (phi>eps)
    H=1;
  else if (phi<-eps)
    H=-1;
  else
    H=phi/eps;
  mu=mu_fluid*(1+H)/2.+mu_air*(1-H)/2.;
}

template<int dim>
void MultiPhase<dim>::get_body_velocity(MappingQ<dim> &mapping, 
					Point<dim> &XG, Vector<double> &VG,
					double &thetaG, double &omegaG,
					double time_step,
					PETScWrappers::MPI::Vector &fluid_velocity_u_Q2,
					PETScWrappers::MPI::Vector &fluid_velocity_v_Q2,
					PETScWrappers::MPI::Vector &body_velocity_u_Q1,
					PETScWrappers::MPI::Vector &body_velocity_v_Q1)
{
  double Fx = 0;
  double Fy = 0;
  double M = 0;
  
  const QGauss<dim-1>  face_quadrature_formula(degree_U+1); // center of the face
  FEFaceValues<dim> fe_face_values_U_mesh (mapping,fe_U_mesh,
					   face_quadrature_formula,
					   update_quadrature_points | update_normal_vectors 
					   | update_JxW_values);
  FEFaceValues<dim> fe_face_values_P (mapping,fe_P,
				      face_quadrature_formula,
				      update_values);
  FEFaceValues<dim> fe_face_values_LS (mapping,fe_LS,
				       face_quadrature_formula,
				       update_values);
  FEFaceValues<dim> fe_face_values_U (mapping,fe_U,
				      face_quadrature_formula,
				      update_gradients);
  
  const unsigned int n_face_q_points = face_quadrature_formula.size();

  std::vector<double>  p (n_face_q_points);
  std::vector<double>  phi (n_face_q_points);
  std::vector<Tensor<1,dim> > grad_u (n_face_q_points);
  std::vector<Tensor<1,dim> > grad_v (n_face_q_points);

  const unsigned int   dofs_per_face = fe_U_mesh.dofs_per_face;
  std::vector<unsigned int> local_dof_indices (dofs_per_face);  

  // loop around cells
  typename DoFHandler<dim>::active_cell_iterator
    cell_U_mesh = dof_handler_U_mesh.begin_active(),
    endc_U_mesh = dof_handler_U_mesh.end();
  typename DoFHandler<dim>::active_cell_iterator
    cell_U = dof_handler_U.begin_active(),
    cell_P = dof_handler_P.begin_active(),
    cell_LS = dof_handler_LS.begin_active();
  
  for (; cell_U_mesh!=endc_U_mesh; ++cell_U_mesh, ++cell_U, ++cell_LS, ++cell_P)
    if (cell_U_mesh->is_locally_owned() && cell_U_mesh_at_box[cell_U_mesh])
      for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
        if (cell_U_mesh->face(face)->at_boundary())
          {
	    cell_U_mesh->face(face)->get_dof_indices(local_dof_indices) ;
	    fe_face_values_U_mesh.reinit(cell_U_mesh,face);
	    fe_face_values_U.reinit(cell_U,face);
	    fe_face_values_P.reinit(cell_P,face);
	    fe_face_values_LS.reinit(cell_LS,face);
	    
	    // get functions at quadrature points
	    fe_face_values_U.get_function_gradients(locally_relevant_solution_u,grad_u);
	    fe_face_values_U.get_function_gradients(locally_relevant_solution_v,grad_v);
	    
	    fe_face_values_P.get_function_values(locally_relevant_solution_p,p);
	    fe_face_values_LS.get_function_values(locally_relevant_solution_phi,phi);

	    for (unsigned int q=0; q<n_face_q_points; ++q)
	      {
		const double JxW = fe_face_values_U_mesh.JxW(q);
		double mu = 1;
		get_mu(mu,phi[q]);
		Tensor<1,dim> n = fe_face_values_U_mesh.normal_vector(q);
		double Fx_at_q_point = -p[q]*n[0] + mu*(grad_u[q]*n + (grad_u[q][0]*n[0]+grad_v[q][0]*n[1]));
		double Fy_at_q_point = -p[q]*n[1] + mu*(grad_v[q]*n + (grad_u[q][1]*n[0]+grad_v[q][1]*n[1]));
		Fx += Fx_at_q_point*JxW;
		Fy += Fy_at_q_point*JxW;
		Point<dim> r(fe_face_values_U_mesh.quadrature_point(q)[0] - XG[0],
			     fe_face_values_U_mesh.quadrature_point(q)[1] - XG[1]);
		M += (r[0]*Fx_at_q_point - r[1]*Fy_at_q_point)*JxW;
	      }
	  }
  // sum over processors
  Utilities::MPI::sum(Fx,mpi_communicator);
  Utilities::MPI::sum(Fy,mpi_communicator);
  Utilities::MPI::sum(M,mpi_communicator);
  
  // add effect of gravity
  Fy += box_mass*(-0);

  double deltaX = time_step*VG[0]+std::pow(time_step,2)*Fx/2/box_mass;
  double deltaY = time_step*VG[1]+std::pow(time_step,2)*Fy/2/box_mass;
  double I = box_mass/6*std::pow(box_width,2);
  double deltaTheta = time_step*omegaG + std::pow(time_step,2)*M/2/I;
  
  pcout << "deltaX: " << deltaX << ", deltaY: " << deltaY << ", deltaTheta: " << deltaTheta << std::endl;
 
  // update velocities and positions
  XG[0] += deltaX;
  XG[1] += deltaY;
  thetaG += deltaTheta;
  
  VG[0] += time_step/box_mass*Fx;
  VG[1] += time_step/box_mass*Fy;
  omegaG += time_step/I*M;

  // GET LOCATION OF BOX BOUNDARY
  Vector<double> Pnp1(2);
  FullMatrix<double> RotMatrix(2,2);
  RotMatrix(0,0)=std::cos(deltaTheta);
  RotMatrix(0,1)=std::sin(deltaTheta);
  RotMatrix(1,0)=-std::sin(deltaTheta);
  RotMatrix(1,1)=std::cos(deltaTheta);

  IndexSet::ElementIterator idofs_iter = locally_owned_dofs_U_mesh.begin();
  for (;idofs_iter!=locally_owned_dofs_U_mesh.end(); idofs_iter++)
    { 
      unsigned int gi = *idofs_iter;
      if (is_dof_at_box[gi]) 
	{	  
	  // Translate
	  x_new_coord(gi) = x_old_coord(gi) + deltaX;
	  y_new_coord(gi) = y_old_coord(gi) + deltaY;
	  
	  // Rotate
	  Vector<double> Pn(2);
	  Pn(0) = x_new_coord(gi);
	  Pn(1) = y_new_coord(gi);
	  RotMatrix.vmult(Pnp1,Pn);
	  // set new coordinates
	  x_new_coord(gi) = Pnp1(0);
	  y_new_coord(gi) = Pnp1(1);
	}
      else
	{
	  x_new_coord(gi) = time_step*fluid_velocity_u_Q2(global_Q1_to_Q2_map[gi])+x_old_coord(gi);
	  y_new_coord(gi) = time_step*fluid_velocity_v_Q2(global_Q1_to_Q2_map[gi])+y_old_coord(gi);
	}
    }

  /////
  // velocity = (smoothed_new_coord - old_coord)/time_step
  idofs_iter = locally_owned_dofs_U_mesh.begin();  
  for (;idofs_iter!=locally_owned_dofs_U_mesh.end(); idofs_iter++)
    {
      int gi = *idofs_iter;
      completely_distributed_mesh_velocity_u_Q1(gi) = (x_new_coord(gi) - x_old_coord(gi))/time_step;
      completely_distributed_mesh_velocity_v_Q1(gi) = (y_new_coord(gi) - y_old_coord(gi))/time_step;
    }
  completely_distributed_mesh_velocity_u_Q1.compress(VectorOperation::insert);
  completely_distributed_mesh_velocity_v_Q1.compress(VectorOperation::insert);
  body_velocity_u_Q1 = completely_distributed_mesh_velocity_u_Q1;
  body_velocity_v_Q1 = completely_distributed_mesh_velocity_v_Q1;
  /////
}

template <int dim>
void MultiPhase<dim>::run()
{
  ////////////////////////
  // GENERAL PARAMETERS //
  ////////////////////////
  umax=1;
  cfl=0.1;
  verbose = true;
  get_output = true;
  output_number = 0;
  n_refinement=4;
  output_time = 0.01;
  final_time = 5.0;
  //////////////////////////////////////////////
  // PARAMETERS FOR THE NAVIER STOKES PROBLEM //
  //////////////////////////////////////////////
  rho_fluid = 100.;
  rho_air = 1.0;
  mu_fluid = 1.0;
  mu_air = 1.0e-2;
  PROBLEM=FLOATING;
  box_width = 0.3;
  box_mass = 4.0;
  
  ForceTerms<dim> force_function(PROBLEM);
  //////////////////////////////////////
  // PARAMETERS FOR TRANSPORT PROBLEM //
  //////////////////////////////////////
  cK = 1.0;
  cE = 1.0;
  sharpness_integer=1; //this will be multipled by min_h
  TRANSPORT_TIME_INTEGRATION=FORWARD_EULER;
  //ALGORITHM = "MPP_u1";
  //ALGORITHM = "NMPP_uH";
  ALGORITHM = "MPP_uH";
  
  //////////////
  // GEOMETRY //
  //////////////
  if (PROBLEM==SMALL_WAVE_PERTURBATION)
    {
      std::vector< unsigned int > repetitions;
      repetitions.push_back(1);
      repetitions.push_back(1);
      GridGenerator::subdivided_hyper_rectangle 
	(triangulation, repetitions, Point<dim>(0.0,0.0), Point<dim>(1.0,1.0), true);
      triangulation.refine_global (n_refinement);
    }
  else  //BOX or FLOATING
    {
      //GridIn<2> gridin;
      //gridin.attach_triangulation(triangulation);
      //std::ifstream f("mesh_box.msh");
      //gridin.read_msh(f);
      //triangulation.refine_global (0);
      make_structured_box(triangulation);
      triangulation.refine_global(n_refinement);
    }

  // SETUP
  setup();
  /////////////
  // MAPPING //
  /////////////
  //MappingQ<dim> mapping_tnp1(1);
  //MappingQ<dim> mapping_tn(1);
  //MappingQ<dim> mapping_tnm1(1);
  Vector<double> mesh_disp_tnp1(dof_handler_U_disp_field.n_dofs()); mesh_disp_tnp1 = 0;
  Vector<double> mesh_disp_tn(dof_handler_U_disp_field.n_dofs()); mesh_disp_tn = 0;
  Vector<double> mesh_disp_tnm1(dof_handler_U_disp_field.n_dofs()); mesh_disp_tnm1 = 0;
  Vector<double> old_coord(dof_handler_U_disp_field.n_dofs()); old_coord = 0;
  Vector<double> new_coord(dof_handler_U_disp_field.n_dofs()); new_coord = 0;
  MappingQEulerian<dim,Vector<double>,dim> mapping_tnp1(1,dof_handler_U_disp_field,mesh_disp_tnp1);
  MappingQEulerian<dim,Vector<double>,dim> mapping_tn(1,dof_handler_U_disp_field,mesh_disp_tn);
  MappingQEulerian<dim,Vector<double>,dim> mapping_tnm1(1,dof_handler_U_disp_field,mesh_disp_tnm1);


  // PARAMETERS FOR TIME STEPPING
  min_h = GridTools::minimal_cell_diameter(triangulation)/std::sqrt(2);
  time_step = cfl*min_h/umax;
  MIN_time_step = 1E-5;
  MAX_time_step = 10*time_step;
  eps=4*min_h; //For reconstruction of density in Navier Stokes
  sharpness=sharpness_integer*min_h; //adjust value of sharpness (for init cond of phi)
  
  // INITIAL CONDITIONS
  initial_condition(mapping_tn);
  output_results(mapping_tn);

  // INITIAL CENTER OF GRAVITY and VELOCITY OF BOX
  Point<dim> XG(0.5,0.5);
  Vector<double> VG(dim);
  VG(0)=0; VG(1)=0;
  double thetaG = 0.;
  double omegaG = 0.;

  /////////////////
  // MOVING MESH //
  /////////////////
  //MESH_VELOCITY = SINUSOIDAL_WITH_FIXED_BOUNDARY;
  MESH_VELOCITY = ZERO_VELOCITY;
  compute_vertex_coord(old_coord); // initial location of vertices

  // NAVIER STOKES SOLVER
  NavierStokesSolver<dim> navier_stokes (degree_LS,
					 degree_U,
					 time_step,
					 eps,
					 rho_air,
					 mu_air,
					 rho_fluid,
					 mu_fluid,
					 force_function,
					 verbose,
					 triangulation,
					 mapping_tnp1, 
					 mapping_tn, 
					 mapping_tnm1,
					 mpi_communicator);
  //set INITIAL CONDITION within NAVIER STOKES
  navier_stokes.initial_condition(locally_relevant_solution_phi,
				  locally_relevant_solution_u,
				  locally_relevant_solution_v,
				  locally_relevant_solution_p);

  // TRANSPORT SOLVER
  LevelSetSolver<dim> level_set (degree_LS,
				 degree_U,
				 time_step,
				 cK,
				 cE, 
				 verbose, 
				 ALGORITHM,
				 TRANSPORT_TIME_INTEGRATION,
				 triangulation, 
				 mapping_tn, 
				 mapping_tnm1,
				 mpi_communicator); 
  // BOUNDARY CONDITIONS FOR PHI
  get_boundary_values_phi(mapping_tn,boundary_values_id_phi,boundary_values_phi);
  level_set.set_boundary_conditions(boundary_values_id_phi,boundary_values_phi);
  
  //set INITIAL CONDITION within TRANSPORT PROBLEM
  level_set.initial_condition(locally_relevant_solution_phi,
			      locally_relevant_solution_u,
			      locally_relevant_solution_v);
  int dofs_U = 2*dof_handler_U.n_dofs();
  int dofs_P = 2*dof_handler_P.n_dofs();
  int dofs_LS = dof_handler_LS.n_dofs();
  int dofs_TOTAL = dofs_U+dofs_P+dofs_LS;
  
  // NO BOUNDARY CONDITIONS for LEVEL SET
  pcout << "Cfl: " << cfl << "; umax: " << umax << "; min h: " << min_h 
	<< "; time step: " << time_step << std::endl;
  pcout << "   Number of active cells:       " 
	<< triangulation.n_global_active_cells() << std::endl
	<< "   Number of degrees of freedom: " << std::endl
	<< "      U: " << dofs_U << std::endl
	<< "      P: " << dofs_P << std::endl
	<< "      LS: " << dofs_LS << std::endl
	<< "      TOTAL: " << dofs_TOTAL
	<< std::endl;
  
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
      //get_interpolated_mesh_velocity(mapping_tn); // From function. Vector in Q2
      //locally_relevant_mesh_velocity_u_Q2 = locally_relevant_solution_u; // Lagrangian
      //locally_relevant_mesh_velocity_v_Q2 = locally_relevant_solution_v;
      //locally_relevant_mesh_velocity_u_Q2=0;// Eulerian 
      //locally_relevant_mesh_velocity_v_Q2=0;//
      get_body_velocity(mapping_tn,XG,VG,thetaG,omegaG,time_step,
			locally_relevant_solution_u, // in Q2
			locally_relevant_solution_v, 
			locally_relevant_body_velocity_u_Q1, // in Q1
			locally_relevant_body_velocity_v_Q1);
      std::cout << "********************" 
		<< locally_relevant_body_velocity_v_Q1.max() << std::endl;
      std::cout << "********************" 
		<< locally_relevant_solution_v.max() << std::endl;
      // interpolate body velocity from Q1 to Q2 to pass it to smoothing process
      interpolate_from_Q1_to_Q2(locally_relevant_body_velocity_u_Q2,locally_relevant_body_velocity_u_Q1);
      interpolate_from_Q1_to_Q2(locally_relevant_body_velocity_v_Q2,locally_relevant_body_velocity_v_Q1);
      get_smoothed_velocity_via_gravity_center(time_step, old_coord, 
					       locally_relevant_body_velocity_u_Q2, //in Q2
					       locally_relevant_body_velocity_v_Q2,
					       locally_relevant_mesh_velocity_u_Q1, //in Q1
					       locally_relevant_mesh_velocity_v_Q1);
      // interpolate mesh velocity from Q1 to Q2: to pass it to Level Set and Navier Stokes solvers
      interpolate_from_Q1_to_Q2(locally_relevant_mesh_velocity_u_Q2,locally_relevant_mesh_velocity_u_Q1);
      interpolate_from_Q1_to_Q2(locally_relevant_mesh_velocity_v_Q2,locally_relevant_mesh_velocity_v_Q1);
      // Pass Q2 mesh velocity to LS and NS solvers
      level_set.set_mesh_velocity(locally_relevant_mesh_velocity_u_Q2, locally_relevant_mesh_velocity_v_Q2);
      navier_stokes.set_mesh_velocity(locally_relevant_mesh_velocity_u_Q2, locally_relevant_mesh_velocity_v_Q2);
      
      ///////////////
      // MOVE MESH // compute mapping tnp1 (using vel at tn)
      ///////////////
      // save old mesh displacements; i.e., save old mappings
      mesh_disp_tnm1=mesh_disp_tn;
      mesh_disp_tn = mesh_disp_tnp1;
      compute_mesh_displacement_via_vectors_in_Q1(time_step, 
						  locally_relevant_mesh_velocity_u_Q1,
						  locally_relevant_mesh_velocity_v_Q1,
						  old_coord, new_coord, mesh_disp_tnp1);
      old_coord.equ(1.0,new_coord);
      
      ///////////////////////////////////
      // ADJUST VELOCITY FOR LEVEL SET // (Physical velocity - mesh velocity)
      ///////////////////////////////////
      completely_distributed_solution_u.equ(1.0,locally_relevant_solution_u);
      completely_distributed_solution_v.equ(1.0,locally_relevant_solution_v);
      completely_distributed_solution_u.add(-1.0,locally_relevant_mesh_velocity_u_Q2);
      completely_distributed_solution_v.add(-1.0,locally_relevant_mesh_velocity_v_Q2);
      locally_relevant_solution_u=completely_distributed_solution_u;
      locally_relevant_solution_v=completely_distributed_solution_v;

      ////////////////////////////
      // GET LEVEL SET SOLUTION //
      ////////////////////////////
      level_set.set_velocity(locally_relevant_solution_u,locally_relevant_solution_v);
      level_set.nth_time_step(time_step);
      level_set.get_unp1(locally_relevant_solution_phi);

      ////////////////////////////////
      // GET NAVIER STOKES VELOCITY //
      ////////////////////////////////
      navier_stokes.set_phi(locally_relevant_solution_phi);

      // boundary conditions for NS (at tnp1)
      get_boundary_values_U(mapping_tnp1,time+time_step);
      navier_stokes.set_boundary_conditions(boundary_values_id_u, boundary_values_id_v,
					    boundary_values_u, boundary_values_v);
      navier_stokes.nth_time_step(time_step);
      navier_stokes.get_velocity(locally_relevant_solution_u,locally_relevant_solution_v);
                 
      /////////////////
      // UPDATE TIME //
      /////////////////
      time+=time_step; // time tnp1

      if (get_output && time-(output_number)*output_time > -1E-10)
	output_results(mapping_tnp1);
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
	unsigned int degree_LS = 1;
	unsigned int degree_U = 2;
        MultiPhase<2> multi_phase(degree_LS, degree_U);
        multi_phase.run();
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
