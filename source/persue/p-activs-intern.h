#ifndef P_ACTIVS_INTERN_H
#define P_ACTIVS_INTERN_H

extern float* cross_entropy_derivs(float* derivs, const float* nodes, const float* targets, size_t amount);

extern float* activ_derivs_apply(float* derivs, const float* values, size_t amount, activ_t activ);

extern float* activ_values(float* result, const float* values, size_t amount, activ_t activ);

#endif // P_ACTIVS_INTERN_H
