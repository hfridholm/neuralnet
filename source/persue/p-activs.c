 #include "../persue.h"

static float sigmoid_value(float value)
{
  return (1 / (1 + exp(-value)) );
}

static float sigmoid_deriv(float value)
{
  return (value * (1 - value));
}

static float relu_value(float value)
{
  return (value > 0) ? value : 0;
}

static float relu_deriv(float value)
{
  return (value > 0) ? 1 : 0;
}

static float tanh_value(float value)
{
  return (exp(2 * value) - 1) / (exp(2 * value) + 1);
}

static float tanh_deriv(float value)
{
  return (1 - value * value);
}

/*
 * RETURN (float* result)
 * - SUCCESS | float* result
 * - ERROR   | NULL
 */
static float* softmax_values(float* result, const float* values, size_t amount)
{
  if(result == NULL || values == NULL) return NULL;

  float sum = 0.0f;

  for(size_t index = 0; index < amount; index++)
  {
    sum += exp(values[index]);
  }
  for(size_t index = 0; index < amount; index++)
  {
    result[index] = exp(values[index]) / sum;
  }
  return result;
}

static float* sigmoid_values(float* result, const float* values, size_t amount)
{
  if(result == NULL || values == NULL) return NULL;

  for(size_t index = 0; index < amount; index++)
  {
    result[index] = sigmoid_value(values[index]);
  }
  return result;
}

static float* relu_values(float* result, const float* values, size_t amount)
{
  if(result == NULL || values == NULL) return NULL;

  for(size_t index = 0; index < amount; index++)
  {
    result[index] = relu_value(values[index]);
  }
  return result;
}

static float* tanh_values(float* result, const float* values, size_t amount)
{
  if(result == NULL || values == NULL) return NULL;

  for(size_t index = 0; index < amount; index++)
  {
    result[index] = tanh_value(values[index]);
  }
  return result;
}

static float** softmax_derivs(float** result, const float* values, size_t amount)
{
  if(result == NULL || values == NULL) return NULL;

  for(size_t iIndex = 0; iIndex < amount; iIndex++)
  {
    for(size_t jIndex = 0; jIndex < amount; jIndex++)
    {
      if(iIndex == jIndex) result[iIndex][jIndex] = values[iIndex] * (1 - values[iIndex]);

      else result[iIndex][jIndex] = -values[iIndex] * values[jIndex];
    }
  }
  return result;
}

/*
 * Apply the derivatives of the softmax activation function
 */
static float* softmax_derivs_apply(float* result, const float* values, size_t amount)
{
  if(result == NULL || values == NULL) return NULL;
 
  float activDerivs[amount][amount];

  softmax_derivs((float**) activDerivs, values, amount);

  float_matrix_vector_dotprod(result, (float**) activDerivs, amount, amount, values);

  return result;
}

 /*
  * Apply the derivatives of the sigmoid activation function
  */
static float* sigmoid_derivs_apply(float* derivs, const float* values, size_t amount)
{
  if(derivs == NULL || values == NULL) return NULL;

  float activDerivs[amount];

  for(size_t index = 0; index < amount; index++)
  {
    activDerivs[index] = sigmoid_deriv(values[index]);
  }

  float_vector_elem_multi(derivs, derivs, activDerivs, amount);

  return derivs;
}

static float* relu_derivs_apply(float* derivs, const float* values, size_t amount)
{
  if(derivs == NULL || values == NULL) return NULL;

  float activDerivs[amount];

  for(size_t index = 0; index < amount; index++)
  {
    activDerivs[index] = relu_deriv(values[index]);
  }

  float_vector_elem_multi(derivs, derivs, activDerivs, amount);

  return derivs;
}

static float* tanh_derivs_apply(float* derivs, const float* values, size_t amount)
{
  if(derivs == NULL || values == NULL) return NULL;

  float activDerivs[amount];

  for(size_t index = 0; index < amount; index++)
  {
    activDerivs[index] = tanh_deriv(values[index]);
  }

  float_vector_elem_multi(derivs, derivs, activDerivs, amount);

  return derivs;
}

/*
 * RETURN (float* result)
 * - SUCCESS | float* result
 * - ERROR   | NULL
 */
float* activ_values(float* result, const float* values, size_t amount, activ_t activ)
{
  switch(activ)
  {
    case ACTIV_NONE: return NULL;

    case ACTIV_SIGMOID: return sigmoid_values(result, values, amount);

    case ACTIV_RELU: return relu_values(result, values, amount);

    case ACTIV_TANH: return tanh_values(result, values, amount);

    case ACTIV_SOFTMAX: return softmax_values(result, values, amount);

    default: return NULL;
  }
}

float* activ_derivs_apply(float* derivs, const float* values, size_t amount, activ_t activ)
{
  switch(activ)
  {
    case ACTIV_NONE: return NULL;

    case ACTIV_SIGMOID: return sigmoid_derivs_apply(derivs, values, amount);

    case ACTIV_RELU: return relu_derivs_apply(derivs, values, amount);

    case ACTIV_TANH: return tanh_derivs_apply(derivs, values, amount);

    case ACTIV_SOFTMAX: return softmax_derivs_apply(derivs, values, amount);

    default: return NULL;
  }
}

float* cross_entropy_derivs(float* derivs, const float* nodes, const float* targets, size_t amount)
{
  if(derivs == NULL || nodes == NULL || targets == NULL) return NULL;
 
  for(size_t index = 0; index < amount; index++)
  {
    derivs[index] = 2 * (nodes[index] - targets[index]);
  }
  return derivs;
}

float cross_entropy_cost(const float* nodes, const float* targets, size_t amount)
{
  if(nodes == NULL || targets == NULL) return -1.0f;

  float cost = 0.0f;

  for(size_t index = 0; index < amount; index++)
  {
    cost += pow(nodes[index] - targets[index], 2);
  }
  return cost / (float) amount;
}
