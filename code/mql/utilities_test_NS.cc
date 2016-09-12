#define SOLUTION_WITH_U_DOT_N_EQUAL_ZERO 1
///////////////////////////////////////////////////////
//////////// EXACT SOLUTION RHO TO TEST NS ////////////
///////////////////////////////////////////////////////
template <int dim>
class RhoFunction : public Function <dim>
{
public:
  RhoFunction (double t=0) : Function<dim>(){this->set_time(t);}
  virtual double value (const Point<dim>   &p, const unsigned int component=0) const;
  virtual Tensor<1,dim> gradient (const Point<dim> &p, const unsigned int component=1) const;
};
template <int dim>
double RhoFunction<dim>::value (const Point<dim> &p,
					      const unsigned int) const
{
  double t = this->get_time();
  double return_value = 0;
  if (dim==2)
    return_value = std::pow(std::sin(p[0]+p[1]+t),2)+1;
  return return_value;
}
template <int dim>
Tensor<1,dim> RhoFunction<dim>::gradient (const Point<dim> &p, const unsigned int) const
{ // THIS IS USED JUST FOR TESTING NS
  Tensor<1,dim> return_value;
  double t = this->get_time();
  double x = p[0]; double y = p[1];

  return_value[0] = 2*std::sin(x+y+t)*std::cos(x+y+t);
  return_value[1] = 2*std::sin(x+y+t)*std::cos(x+y+t);
  return return_value;
}

template <int dim>
class NuFunction : public Function <dim>
{
public:
  NuFunction (double t=0) : Function<dim>(){this->set_time(t);}
  virtual double value (const Point<dim>   &p, const unsigned int component=0) const;
};
template <int dim>
double NuFunction<dim>::value (const Point<dim> &p, const unsigned int) const
{
  return 1.0;
}

//////////////////////////////////////////////////////////////////
/////////////////// EXACT SOLUTION U to TEST NS //////////////////
//////////////////////////////////////////////////////////////////
template <int dim>
class ExactSolution_and_BC_U : public Function <dim>
{
public:
  ExactSolution_and_BC_U (double t=0, int field=0) 
    : 
    Function<dim>(), 
    field(field)
  {
    this->set_time(t);
  }
  virtual double value (const Point<dim> &p, const unsigned int  component=1) const;
  virtual Tensor<1,dim> gradient (const Point<dim> &p, const unsigned int component=1) const;
  virtual void set_field(int field) {this->field=field;}
  int field;
  unsigned int type_simulation;
};
template <int dim>
double ExactSolution_and_BC_U<dim>::value (const Point<dim> &p,
					   const unsigned int) const
{
  double t = this->get_time();
  double return_value = 0;
  double Pi = numbers::PI;
  double x = p[0]; double y = p[1]; double z = 0;

  if (dim == 2)
    if (field == 0)
      if (SOLUTION_WITH_U_DOT_N_EQUAL_ZERO)
	return_value = std::sin(2*Pi*x)*std::cos(2*Pi*y)*std::sin(t);
      else 
	return_value = std::sin(x)*std::sin(y+t);
    else
      if (SOLUTION_WITH_U_DOT_N_EQUAL_ZERO)
	return_value = -std::sin(2*Pi*y)*std::cos(2*Pi*x)*std::sin(t);
      else 
	return_value = std::cos(x)*std::cos(y+t); 
  return return_value;  
}

template <int dim>
Tensor<1,dim> ExactSolution_and_BC_U<dim>::gradient (const Point<dim> &p,
						     const unsigned int) const
{ // THIS IS USED JUST FOR TESTING NS
  Tensor<1,dim> return_value;
  double t = this->get_time();
  double Pi = numbers::PI;
  double x = p[0]; double y = p[1]; double z = 0;
  if (dim == 2)
    if (field == 0)
      {
	if (SOLUTION_WITH_U_DOT_N_EQUAL_ZERO)
	  {
	    return_value[0] = 2*Pi*std::cos(2*Pi*x)*std::cos(2*Pi*y)*std::sin(t);
	    return_value[1] = -2*Pi*std::sin(t)*std::sin(2*Pi*x)*std::sin(2*Pi*y);
	  }
	else
	  {
	    return_value[0] = std::cos(x)*std::sin(y+t);
	    return_value[1] = std::sin(x)*std::cos(y+t);
	  }
      }
    else 
      {
	if (SOLUTION_WITH_U_DOT_N_EQUAL_ZERO)
	  {
	    return_value[0] = 2*Pi*std::sin(t)*std::sin(2*Pi*x)*std::sin(2*Pi*y);
	    return_value[1] = -2*Pi*std::cos(2*Pi*x)*std::cos(2*Pi*y)*std::sin(t);
	  }
	else 
	  {
	    return_value[0] = -std::sin(x)*std::cos(y+t);
	    return_value[1] = -std::cos(x)*std::sin(y+t);
	  }
      }
  return return_value;
}

///////////////////////////////////////////////////////
/////////// EXACT SOLUTION FOR p TO TEST NS ///////////
///////////////////////////////////////////////////////
template <int dim>
class ExactSolution_p : public Function <dim>
{
public:
  ExactSolution_p (double t=0) : Function<dim>(){this->set_time(t);}
  virtual double value (const Point<dim> &p, const unsigned int  component=0) const;
  virtual Tensor<1,dim> gradient (const Point<dim> &p, const unsigned int component = 0) const;
};

template <int dim>
double ExactSolution_p<dim>::value (const Point<dim> &p, const unsigned int) const
{
  double t = this->get_time();
  double return_value = 0;
  if (dim == 2)
    return_value = std::cos(p[0])*std::sin(p[1]+t)-std::sin(1)*(std::cos(t)-cos(1+t));
  return return_value;
}

template <int dim>
Tensor<1,dim> ExactSolution_p<dim>::gradient (const Point<dim> &p, const unsigned int) const
{
  Tensor<1,dim> return_value;
  double t = this->get_time();
  if (dim == 2)
    {
      return_value[0] = -std::sin(p[0])*std::sin(p[1]+t);
      return_value[1] = std::cos(p[0])*std::cos(p[1]+t);
    }
  return return_value;
}

//////////////////////////////////////////////////////////////////
//////////////////// FORCE TERMS to TEST NS //////////////////////
//////////////////////////////////////////////////////////////////
template <int dim>
class ForceTerms : public Function <dim>
{
public:
  ForceTerms (double t=0) 
    : 
    Function<dim>() 
  {
    this->set_time(t);
    mu = 1.0;
  }
  virtual void vector_value (const Point<dim> &p, Vector<double> &values) const;
  double mu;
};

template <int dim>
void ForceTerms<dim>::vector_value (const Point<dim> &p, Vector<double> &values) const
{ 
  double x = p[0]; double y = p[1]; double z = 0;
  double t = this->get_time();
  double Pi = numbers::PI;
  
  if (dim == 2)
    {
      if (SOLUTION_WITH_U_DOT_N_EQUAL_ZERO)
	{
	  // force in x
	  values[0] = 
	    std::cos(2*Pi*y)*std::sin(2*Pi*x)*(std::cos(t)*(1+std::pow(std::sin(t+x+y),2))+std::sin(t)*std::sin(2*(t+x+y))) // time derivative
	    +8*mu*std::pow(Pi,2)*std::cos(2*Pi*y)*std::sin(t)*std::sin(2*Pi*x) //viscosity 
	    +(std::pow(std::sin(t),2)*std::sin(2*Pi*x) //nonlinearity 
	    *(8*std::pow(std::cos(2*Pi*y),2)*std::cos(t+x+y)*std::sin(2*Pi*x)*std::sin(t+x+y)- 
	    2*std::cos(2*Pi*x)*(2*Pi*(-3+std::cos(2*(t+x+y)))+std::sin(4*Pi*y)*std::sin(2*(t+x+y)))))/4.	
	    -std::sin(x)*std::sin(t+y);// pressure
	  // force in y 
	  values[1] = 
	    -(std::cos(2*Pi*x)*std::sin(2*Pi*y)*(std::cos(t)*(1+std::pow(std::sin(t+x+y),2))+std::sin(t)*std::sin(2*(t+x+y)))) //time derivative
	    -8*mu*std::pow(Pi,2)*std::cos(2*Pi*x)*std::sin(t)*std::sin(2*Pi*y) // viscosity
	    +(std::pow(std::sin(t),2)*std::sin(2*Pi*y) // nonlinearity
	    *(-2*Pi*std::cos(2*Pi*y)*(-3+std::cos(2*(t+x+y)))
	    -2*std::cos(2*Pi*x)*std::sin(2*Pi*(x-y))*std::sin(2*(t+x+y))))/2. 
	    +std::cos(x)*std::cos(t+y); // pressure
	}
      else
	{
	  // force in x
	  values[0] = 
	    std::sin(x)*(std::cos(t+y)*(1+std::pow(std::sin(t+x+y),2))+std::sin(t+y)*std::sin(2*(t+x+y)))
	    +2*mu*std::sin(x)*std::sin(t+y) // viscosity
	    +(std::sin(x)*std::sin(t+y)*(std::sin(t+3*x+y)+std::sin(3*t+x+3*y)))/2.
	    -((-3+std::cos(2*(t+x+y)))*std::sin(2*x))/4. //non-linearity
	    -std::sin(x)*std::sin(y+t); // pressure
	  // force in y 
	  values[1] = 
	    (std::cos(x)*(-6*std::sin(t+y)+std::sin(t+2*x+y)+3*std::sin(3*t+2*x+3*y)))/4.
	    +2*mu*std::cos(x)*std::cos(t+y) // viscosity
	    +(std::cos(x)*std::cos(t+y)*(std::sin(t+3*x+y)+std::sin(3*t+x+3*y)))/2.
	    +((-3+std::cos(2*(t+x+y)))*std::sin(2*(t+y)))/4. //non-linearity
	    +std::cos(x)*std::cos(y+t); // pressure
	}
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
    return std::sin(pi*x)*std::cos(pi*y)*std::cos(2*pi*time);
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
    return -std::cos(pi*x)*std::sin(pi*y)*std::cos(2*pi*time);      
  else //ZERO VELOCITY
    return 0.;
}
