#include "../secure.h"

/*
 * RETURN
 * - SUCCESS | The filtered matrix result
 * - ERROR   | NULL
 */
float** float_matrix_filter_index(float** result, float** matrix, size_t height, size_t width, const int* indexes, size_t amount)
{
  if(result == NULL || matrix == NULL || indexes == NULL) return NULL;

  for(size_t index = 0; index < amount; index++)
  {
    size_t wIndex = indexes[index];

    if(wIndex < 0 || wIndex >= width) return NULL;

    for(size_t hIndex = 0; hIndex < height; hIndex++)
    {
      result[hIndex][index] = matrix[hIndex][wIndex];
    }
  }
  return result;
}

/*
 * Create a float matrix allocated on the HEAP using malloc
 *
 * RETURN
 * - SUCCESS | The created float matrix
 * - ERROR   | NULL
 */
float** float_matrix_create(size_t height, size_t width)
{
  if(height <= 0 || width <= 0) return NULL;

  float** matrix = malloc(sizeof(float*) * height);

  if(matrix == NULL) return NULL;

  for(size_t index = 0; index < height; index++)
  {
    matrix[index] = float_vector_create(width);
  }
  return matrix;
}

/*
 * Free a float matrix from the HEAP using free
 * Also assigns NULL to pointer
 */
void float_matrix_free(float*** matrix, size_t height, size_t width)
{
  if(*matrix == NULL) return;

  for(size_t index = 0; index < height; index++)
  {
    float_vector_free((*matrix) + index, width);
  }
  free(*matrix);

  *matrix = NULL;
}

/*
 * Create a matrix with random values
 *
 * RETURN
 * - SUCCESS | Matrix with random values
 * - ERROR   | NULL
 */
float** float_matrix_random_create(size_t height, size_t width, float min, float max)
{
  if(height <= 0 || width <= 0) return NULL;

  float** matrix = malloc(sizeof(float*) * height);

  if(matrix == NULL) return NULL;

  for(size_t index = 0; index < height; index++)
  {
    matrix[index] = float_vector_random_create(width, min, max);
  }
  return matrix;
}

/*
 * Transpose matrix by flipping it over the downwards diagonal
 *
 * RETURN
 * - SUCCESS | The transposed matrix
 * - ERROR   | NULL
 */
float** float_matrix_transp(float** transp, float** matrix, size_t height, size_t width)
{
  if(transp == NULL || matrix == NULL) return NULL;

  for(size_t hIndex = 0; hIndex < height; hIndex++)
  {
    for(size_t wIndex = 0; wIndex < width; wIndex++)
    {
      transp[wIndex][hIndex] = matrix[hIndex][wIndex];
    }
  }
  return transp;
}

/*
 * Mulitply matrix values by a scalor
 *
 * RETURN
 * - SUCCESS | The scaled matrix
 * - ERROR   | NULL
 */
float** float_matrix_scale_multi(float** result, float** matrix, size_t height, size_t width, float scalor)
{
  if(result == NULL || matrix == NULL) return NULL;

  for(size_t hIndex = 0; hIndex < height; hIndex++)
  {
    for(size_t wIndex = 0; wIndex < width; wIndex++)
    {
      result[hIndex][wIndex] = (matrix[hIndex][wIndex] * scalor);
    }
  }
  return result;
}

/*
 * Add the values of two matricies together with each other
 *
 * RETURN
 * - SUCCESS | float** result
 * - ERROR   | NULL
 */
float** float_matrix_elem_addit(float** result, float** matrix1, float** matrix2, size_t height, size_t width)
{
  if(result == NULL || matrix1 == NULL || matrix2 == NULL) return NULL;

  for(size_t hIndex = 0; hIndex < height; hIndex++)
  {
    for(size_t wIndex = 0; wIndex < width; wIndex++)
    {
      result[hIndex][wIndex] = (matrix1[hIndex][wIndex] + matrix2[hIndex][wIndex]);
    }
  }
  return result;
}

/*
 * Return the dot product of a matrix and a vector with different lengths
 *
 * Note: Maybe just return either result or NULL
 *
 * RETURN
 * - 0 | Success!
 * - 1 | An inputted pointer was NULL
 *
 */
int float_matrix_vector_dotprod(float* result, float** matrix, size_t height, size_t width, const float* vector)
{
  if(result == NULL || matrix == NULL || vector == NULL) return 1;

  float tresult[height];
  memset(tresult, 0.0f, sizeof(float) * height);
 
  for(size_t hIndex = 0; hIndex < height; hIndex++)
  {
    for(size_t wIndex = 0; wIndex < width; wIndex++)
    {
      tresult[hIndex] += (matrix[hIndex][wIndex] * vector[wIndex]);
    }
  }
  float_vector_copy(result, tresult, height);

  return 0;
}

/*
 * Copy the content of source to destin
 *
 * RETURN
 * - SUCCESS | The pointer to the destination matrix
 * - ERROR   | NULL
 */
float** float_matrix_copy(float** destin, float** source, size_t height, size_t width)
{
  if(destin == NULL || source == NULL) return NULL;

  for(size_t index = 0; index < height; index++)
  {
    destin[index] = float_vector_copy(destin[index], source[index], width);
  }
  return destin;
}

/*
 * Print the inputted matrix to the console
 */
void float_matrix_print(float** matrix, size_t height, size_t width)
{
  if(matrix == NULL) return;

  for(size_t index = 0; index < height; index++)
  {
    float_vector_print(matrix[index], width);
  }
}
