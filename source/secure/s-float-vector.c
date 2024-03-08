#include "../secure.h"

/*
 * Searches the min and max values of a vector
 *
 * RETURN
 * - 0 | Success!
 * - 1 | Failed becuase vector is NULL
 * - 1 | Failed because vector has no elements
 */
int float_vector_minmax(float* min, float* max, const float* vector, size_t length)
{
  if(vector == NULL) return 1;

  // Maybe just return 1 on error
  if(length <= 0) return 2;

  *min = vector[0];
  *max = vector[0];

  for(size_t index = 1; index < length; index++)
  {
    if(vector[index] > *max) *max = vector[index];

    if(vector[index] < *min) *min = vector[index];
  }
  return 0; // Success!
}

/*
 * Returns a random float between min and max
 */
float float_random_create(float min, float max)
{
  float fraction = ((float) rand() / (float) RAND_MAX);

  return (fraction * (max - min) + min);
}

/*
 * Create a float vector allocated on the HEAP using malloc
 * Also clean the memory using memset
 *
 * RETURN
 * - SUCCESS | The created float vector
 * - ERROR   | NULL
 */
float* float_vector_create(size_t length)
{
  if(length <= 0) return NULL;

  float* vector = malloc(sizeof(float) * length);

  if(vector == NULL) return NULL;

  memset(vector, 0.0f, sizeof(float) * length);

  return vector;
}

/*
 * Free a float vector from the HEAP using free
 * Also assigns NULL to pointer
 */
void float_vector_free(float** vector, size_t length)
{
  free(*vector);

  *vector = NULL;
}

/*
 *
 */
float* float_vector_random_create(size_t length, float min, float max)
{
  if(length <= 0) return NULL;

  float* vector = malloc(sizeof(float) * length);

  if(vector == NULL) return NULL;

  for(size_t index = 0; index < length; index++)
  {
    vector[index] = float_random_create(min, max);
  }
  return vector;
}

/*
 * Fills a vector with random floats between min and max
 * The function modifies the inputted vector
 *
 * RETURN
 * - SUCCESS | The filled vector
 * - ERROR   | NULL
 */
float* float_vector_random_fill(float* vector, size_t length, float min, float max)
{
  if(vector == NULL) return NULL;

  for(size_t index = 0; index < length; index++)
  {
    vector[index] = float_random_create(min, max);
  }
  return vector;
}

/*
 * Scale all the vectors values by a scalor
 *
 * RETURN
 * - SUCCESS | The vector with scaled values
 * - ERROR   | NULL
 */
float* float_vector_scale_multi(float* result, const float* vector, size_t length, float scalor)
{
  if(result == NULL || vector == NULL) return NULL;

  for(size_t index = 0; index < length; index++)
  {
    result[index] = (vector[index] * scalor);
  }
  return result;
}

/*
 * Add a scalor to every value of the inputted vector
 * 
 * RETURN
 * - SUCCESS | float* result
 * - ERROR   | NULL
 */
float* float_vector_scale_addit(float* result, const float* vector, size_t length, float scalor)
{
  if(result == NULL || vector == NULL) return NULL;

  for(size_t index = 0; index < length; index++)
  {
    result[index] = (vector[index] + scalor);
  }
  return result;
}

/*
 * Multiply the values of two vectors with each other
 *
 * RETURN
 * - SUCCESS | The vector with multiplied values
 * - ERROR   | NULL
 */
float* float_vector_elem_multi(float* result, const float* vector1, const float* vector2, size_t length)
{
  if(result == NULL || vector1 == NULL || vector2 == NULL) return NULL;

  for(size_t index = 0; index < length; index++)
  {
    result[index] = (vector1[index] * vector2[index]);
  }
  return result;
}

/*
 * Add the values of two vectors together
 *
 * RETURN
 * - SUCCESS | The vector with added values
 * - ERROR   | NULL
 */
float* float_vector_elem_addit(float* result, const float* vector1, const float* vector2, size_t length)
{
  if(result == NULL || vector1 == NULL || vector2 == NULL) return NULL;

  for(size_t index = 0; index < length; index++)
  {
    result[index] = (vector1[index] + vector2[index]);
  }
  return result;
}

/*
 * Return the dot product of two vectors of different lengths
 *
 * RETURN
 * - SUCCESS | The dot product matrix
 * - ERROR   | NULL
 */
float** float_vector_dotprod(float** result, const float* vector1, size_t length1, const float* vector2, size_t length2)
{
  if(result == NULL || vector1 == NULL || vector2 == NULL) return NULL;

  for(size_t hIndex = 0; hIndex < length1; hIndex++)
  {
    for(size_t wIndex = 0; wIndex < length2; wIndex++)
    {
      result[hIndex][wIndex] = (vector1[hIndex] * vector2[wIndex]);
    }
  }
  return result;
}

/*
 * Create a new vector and copy the values of the inputted vector
 *
 * RETURN
 * - SUCCESS | The newly created duplicated vector
 * - ERROR   | NULL
 */
float* float_vector_duplic(const float* vector, size_t length)
{
  if(vector == NULL || length <= 0) return NULL;

  float* duplic = malloc(sizeof(float) * length);

  if(duplic == NULL) return NULL;

  return memmove(duplic, vector, sizeof(float) * length);
}

/*
 * Copy the content of source to destin, using memmove
 *
 * RETURN (same as memmove)
 * - SUCCESS | The pointer to the desination vector
 * - ERROR   | NULL
 */
float* float_vector_copy(float* destin, const float* source, size_t length)
{
  if(destin == NULL || source == NULL) return NULL;

  return memmove(destin, source, sizeof(float) * length);
}

/*
 * Return the sum of the vectors values
 */
float float_vector_sum(const float* vector, size_t length)
{
  float sum = 0;

  if(vector == NULL) return sum;

  for(size_t index = 0; index < length; index++)
  {
    sum += vector[index];
  }
  return sum;
}

/*
 * Print the inputted vector to the console
 */
void float_vector_print(const float* vector, size_t length)
{
  if(vector == NULL) return;

  for(size_t index = 0; index < length; index++)
  {
    printf("%+04.8f ", vector[index]);
  }
  printf("\n");
}
