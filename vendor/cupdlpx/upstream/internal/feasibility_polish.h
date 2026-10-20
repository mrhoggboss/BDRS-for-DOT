#pragma once

#include "internal_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    void feasibility_polish(const pdhg_parameters_t *params, pdhg_solver_state_t *state);

#ifdef __cplusplus
}
#endif
