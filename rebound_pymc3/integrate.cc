#section support_code_struct

int APPLY_SPECIFIC(integrate)(
    PyArrayObject *input_masses,     // Array of masses (nbody, )
    PyArrayObject *input_coords,     // Array of body coordinates (nbody, 6)
    PyArrayObject *input_times,      // Array of times (ntime, )
    PyArrayObject **output_coords,   // The output coordinates (6, ntime, nbody)
    PyArrayObject **output_jacobian, // The Jacobian (6, ntime, nbody, nbody, 7)
    PARAMS_TYPE *params)
{
  using namespace rebound_pymc3;

  int success = 0;
  npy_intp nbody = -1;
  auto masses = get_input<DTYPE_INPUT_0>(&nbody, input_masses, &success);

  int ndim = 2;
  npy_intp shape[] = {nbody, 6};
  auto initial_coords = get_input<DTYPE_INPUT_1>(ndim, shape, input_coords, &success);

  npy_intp ntime = -1;
  auto times = get_input<DTYPE_INPUT_2>(&ntime, input_times, &success);

  if (success)
    return 1;

  npy_intp final_shape[] = {ntime, nbody, 6};
  auto final_coords = allocate_output<DTYPE_OUTPUT_0>(3, final_shape, TYPENUM_OUTPUT_0, output_coords, &success);

  npy_intp jac_shape[] = {ntime, nbody, 6, nbody, 7};
  auto jacobian = allocate_output<DTYPE_OUTPUT_1>(5, jac_shape, TYPENUM_OUTPUT_1, output_jacobian, &success);

  if (success)
    return 1;

  struct reb_simulation *sim = reb_create_simulation();
  sim->t = params->t;
  sim->dt = params->dt;

  // Set the integrator; enums aren't playing nice between c and c++
  switch (params->integrator)
  {
  case (0):
    sim->integrator = reb_simulation::REB_INTEGRATOR_IAS15;
    break;
  case (1):
    sim->integrator = reb_simulation::REB_INTEGRATOR_WHFAST;
    break;
  case (2):
    sim->integrator = reb_simulation::REB_INTEGRATOR_SEI;
    break;
  case (4):
    sim->integrator = reb_simulation::REB_INTEGRATOR_LEAPFROG;
    break;
  case (7):
    sim->integrator = reb_simulation::REB_INTEGRATOR_NONE;
    break;
  case (8):
    sim->integrator = reb_simulation::REB_INTEGRATOR_JANUS;
    break;
  case (9):
    sim->integrator = reb_simulation::REB_INTEGRATOR_MERCURIUS;
    break;

  default:
    PyErr_Format(PyExc_ValueError, "unknown integrator");
    reb_free_simulation(sim);

    return 1;
  }

  for (npy_intp i = 0; i < nbody; ++i)
  {
    struct reb_particle body = {0.0};
    body.m = double(masses[i]);
    int ind = i * 6;
    body.x = double(initial_coords[ind + 0]);
    body.y = double(initial_coords[ind + 1]);
    body.z = double(initial_coords[ind + 2]);
    body.vx = double(initial_coords[ind + 3]);
    body.vy = double(initial_coords[ind + 4]);
    body.vz = double(initial_coords[ind + 5]);
    reb_add(sim, body);
  }

  for (npy_intp t = 0; t < ntime; ++t)
  {
    reb_integrate(sim, times[t]);

    for (npy_intp i = 0; i < nbody; ++i)
    {
      struct reb_particle particle = sim->particles[i];
      int ind = (t * nbody + i) * 6;
      final_coords[ind + 0] = particle.x;
      final_coords[ind + 1] = particle.y;
      final_coords[ind + 2] = particle.z;
      final_coords[ind + 3] = particle.vx;
      final_coords[ind + 4] = particle.vy;
      final_coords[ind + 5] = particle.vz;
    }
  }

  reb_free_simulation(sim);

  return 0;
}
