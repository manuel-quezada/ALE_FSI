/////////////////////////////////////////////////////
//////////////////// INITIAL PHI ////////////////////
/////////////////////////////////////////////////////
template <int dim>
class InitialPhi : public Function <dim>
{
public:
  InitialPhi (unsigned int PROBLEM, double sharpness=0.005) : Function<dim>(),
							      sharpness(sharpness),
							      PROBLEM(PROBLEM) {}
  virtual double value (const Point<dim> &p, const unsigned int component=0) const;
  double sharpness;
  unsigned int PROBLEM;
};
template <int dim>
double InitialPhi<dim>::value (const Point<dim> &p,
			       const unsigned int) const
{
  double x = p[0]; double y = p[1];
  double pi=numbers::PI;

  if (PROBLEM==SMALL_WAVE_PERTURBATION)
    {
      double wave = 0.1*std::sin(pi*x)+0.5;
      if (y <= wave)
	return 1.0;
      else 
	return -1.0;
    }
  else if (PROBLEM==BOX_PROBLEM || PROBLEM==FLOATING)
    {
      //return -std::atan((y-0.5)/0.1);
      if (y<=0.5)
	return 1.0;
      else 
	return -1.0;
    }
  else
    {
      std::cout << "Error in type of PROBLEM" << std::endl;
      abort();
    }
}

///////////////////////////////////////////////////////
//////////////////// FORCE TERMS ///// ////////////////
///////////////////////////////////////////////////////
template <int dim>
class ForceTerms : public Function <dim>
{
public:
  ForceTerms (unsigned int PROBLEM, double t=0) : Function<dim>(), PROBLEM(PROBLEM)
  {this->set_time(t);}
  virtual void vector_value (const Point<dim> &p, Vector<double> &values) const;
  unsigned int PROBLEM;
};
template <int dim>
void ForceTerms<dim>::vector_value (const Point<dim> &p, Vector<double> &values) const
{ 
  values[0]=0.;
  values[1]=-1.;
}

/////////////////////////////////////////////////////
//////////////////// BOUNDARY PHI ///////////////////
/////////////////////////////////////////////////////
template <int dim>
class BoundaryPhi : public Function <dim>
{
public:
  BoundaryPhi (double t=0) 
    : 
    Function<dim>() 
  {this->set_time(t);}
  virtual double value (const Point<dim> &p, const unsigned int component=0) const;
};
template <int dim>
double BoundaryPhi<dim>::value (const Point<dim> &p,
				const unsigned int) const
{
  // boundary for filling the tank
  return 1.0;
}

///////////////////////////////////////////////////////
/////////////////// POST-PROCESSING ///////////////////
///////////////////////////////////////////////////////
template <int dim>
class Postprocessor : public DataPostprocessorScalar <dim>
{
public:
  Postprocessor(double eps, double rho_air, double rho_fluid)
    :
    DataPostprocessorScalar<dim>("Density",update_values)
  {
    this->eps=eps;
    this->rho_air=rho_air;
    this->rho_fluid=rho_fluid;
  }
  virtual void compute_derived_quantities_scalar (const std::vector< double > &uh,
						  const std::vector< Tensor< 1, dim > > &duh,
						  const std::vector< Tensor< 2, dim > > &dduh,
						  const std::vector< Point< dim > > &normals,
						  const std::vector< Point< dim > > &evaluation_points,
						  std::vector< Vector< double > > &computed_quantities 
						  ) const;
  double eps;
  double rho_air;
  double rho_fluid;
};

template <int dim>
void Postprocessor<dim>::compute_derived_quantities_scalar(const std::vector< double > &uh,
							   const std::vector< Tensor< 1, dim > > & /*duh*/,
							   const std::vector< Tensor< 2, dim > > & /*dduh*/,
							   const std::vector< Point< dim > > & /*normals*/,
							   const std::vector< Point< dim > > & /*evaluation_points*/,
							   std::vector< Vector< double > > &computed_quantities) const
{
  const unsigned int n_quadrature_points = uh.size();
  for (unsigned int q=0; q<n_quadrature_points; ++q)
    {
      double H;
      double rho_value;
      double phi_value=uh[q];
      if(phi_value > eps) 
	H=1;
      else if (phi_value < -eps) 
	H=-1;
      else 
	H=phi_value/eps;
      rho_value = rho_fluid*(1+H)/2. + rho_air*(1-H)/2.;
      computed_quantities[q] = rho_value;
    }
}

//////////////////////////////////////////////////////
//////////////////// MESH VELOCITY ///////////////////
//////////////////////////////////////////////////////
template <int dim>
class MeshVelocityU : public Function<dim>
{
public:
  MeshVelocityU (unsigned int MESH_VELOCITY, double time=0) : Function<dim>(), 
							 MESH_VELOCITY(MESH_VELOCITY), time(time){}
  virtual double value (const Point<dim> &p, const unsigned int component=0) const;
  void set_time(double time){this->time=time;};
  unsigned MESH_VELOCITY;
  double time;
};

template <int dim>
double MeshVelocityU<dim>::value (const Point<dim> &p, const unsigned int) const
{
  double x = p[0]; double y = p[1];
  double pi = numbers::PI;

  if (MESH_VELOCITY==SINUSOIDAL_WITH_FIXED_BOUNDARY)
    return 1.0*std::sin(pi*x)*std::cos(pi*y)*std::cos(2*pi*time);
  else if (MESH_VELOCITY=BOX_VELOCITY)
    return 0;
  else //ZERO VELOCITY
    return 0.;
}

template <int dim>
class MeshVelocityV : public Function<dim>
{
public:
  MeshVelocityV (unsigned int MESH_VELOCITY, double time=0) : Function<dim>(), 
							      MESH_VELOCITY(MESH_VELOCITY), time(time){}
  virtual double value (const Point<dim> &p, const unsigned int component=0) const;
  void set_time(double time){this->time=time;};
  unsigned MESH_VELOCITY;
  double time;
};

template <int dim>
double MeshVelocityV<dim>::value (const Point<dim> &p, const unsigned int) const
{
  double x = p[0]; double y = p[1];
  double pi = numbers::PI;

  if (MESH_VELOCITY==SINUSOIDAL_WITH_FIXED_BOUNDARY)
    return -1.0*std::cos(pi*x)*std::sin(pi*y)*std::cos(2*pi*time);      
  else if (MESH_VELOCITY==BOX_VELOCITY)
    return 0.25*std::sin(2*pi*time);
  else //ZERO VELOCITY
    return 0.;
}
