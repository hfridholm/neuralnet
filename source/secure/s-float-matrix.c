#include "../secure.h"

/*
 * Create a float matrix array allocated on the HEAP using malloc
 *
 * RETURN (float*** matarr)
 * - SUCCESS | The created float matrix array
 * - ERROR   | NULL
 */
float*** float_matarr_create(size_t amount, size_t height, size_t width)
{
  if(amount <= 0 || height <= 0 || width <= 0) return NULL;
 
  float*** matarr = malloc(sizeof(float**) * amount);

  if(matarr == NULL) return NULL;

  for(size_t index = 0; index < amount; index++)
  {
    matarr[index] = float_matrix_create(height, width);
  }
  return matarr;
}

/*
 * Free a float matrix array from the HEAP using free
 * Also assigns NULL to pointer
 */
void float_matarr_free(float**** matarr, size_t amount, size_t height, size_t width)
{
  if(*matarr == NULL) return;

  for(size_t index = 0; index < amount; index++)
  {
    float_matrix_free((*matarr) + index, height, width);
  }
  free(*matarr);

  *matarr = NULL;
}

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
 * RETURN
 * - SUCCESS | The matrix column vector
 * - ERROR   | NULL
 */
float* float_matrix_column_vector(float* vector, float** matrix, size_t height, size_t width, size_t column)
{
  if(vector == NULL || matrix == NULL) return NULL;

  if(column < 0 || column >= width) return NULL;

  for(size_t index = 0; index < height; index++)
  {
    vector[index] = matrix[index][column];
  }
  return vector;
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
 * Fill a matrix with random values
 *
 * RETURN
 * - SUCCESS | The filled matrix with random values
 * - ERROR   | NULL
 */
float** float_matrix_random_fill(float** matrix, size_t height, size_t width, float min, float max)
{
  if(matrix == NULL) return NULL;

  for(size_t index = 0; index < height; index++)
  {
    float_vector_random_fill(matrix[index], width, min, max);
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
 * Add a scalor to every value of the inputted matrix
 *
 * RETURN
 * - SUCCESS | The scaled matrix
 * - ERROR   | NULL
 */
float** float_matrix_scale_addit(float** result, float** matrix, size_t height, size_t width, float scalor)
{
  if(result == NULL || matrix == NULL) return NULL;

  for(size_t index = 0; index < height; index++)
  {
    result[index] = float_vector_scale_addit(result[index], matrix[index], width, scalor);
  }
  return result;
}

/*
 * Multiply the values of two matricies with each other
 *
 * RETURN
 * - SUCCESS | float** result
 * - ERROR   | NULL
 */
float** float_matrix_elem_multi(float** result, float** matrix1, float** matrix2, size_t height, size_t width)
{
  if(result == NULL || matrix1 == NULL || matrix2 == NULL) return NULL;

  for(size_t hIndex = 0; hIndex < height; hIndex++)
  {
    for(size_t wIndex = 0; wIndex < width; wIndex++)
    {
      result[hIndex][wIndex] = (matrix1[hIndex][wIndex] * matrix2[hIndex][wIndex]);
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
 * Add the values of a matrix row with each value of a vector
 *
 * RETURN
 * - SUCCESS | The result matrix
 * - ERROR   | NULL
 */
float** float_matrix_vector_addit(float** result, float** matrix, const float* vector, size_t height, size_t width)
{
  if(result == NULL || matrix == NULL || vector == NULL) return NULL;

  for(size_t hIndex = 0; hIndex < height; hIndex++)
  {
    for(size_t wIndex = 0; wIndex < width; wIndex++)
    {
      result[hIndex][wIndex] = (matrix[hIndex][wIndex] + vector[wIndex]);
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
 
  for(size_t hIndex = 0; hIndex < height; hIndex++)
  {
    for(size_t wIndex = 0; wIndex < width; wIndex++)
    {
      result[hIndex] += (matrix[hIndex][wIndex] * vector[wIndex]);
    }
  }
  return 0;
}

/*
 * Convert vector to matrix by assigning each value on new column
 *
 * RETURN
 * - SUCCESS | The converted matrix
 * - ERROR   | NULL
 */
float** float_vector_cnvrt_matrix(float** matrix, const float* vector, size_t length)
{
  if(matrix == NULL || vector == NULL) return NULL;

  for(size_t index = 0; index < length; index++)
  {
    matrix[index][0] = vector[index];
  }
  return matrix;
}

/*
 * Assign matrix values to a vector starting from index 0
 *
 * RETURN
 * - SUCCESS | The converted vector
 * - ERROR   | NULL
 */
float* float_matrix_cnvrt_vector(float* vector, float** matrix, size_t height, size_t width)
{
  if(vector == NULL || matrix == NULL) return NULL;

  for(size_t hIndex = 0; hIndex < height; hIndex++)
  {
    for(size_t wIndex = 0; wIndex < width; wIndex++)
    {
      vector[hIndex * width + wIndex] = matrix[hIndex][wIndex];
    }
  }
  return vector;
}

/*
 * Create a new matrix and copy the values of the inputted matrix
 *
 * RETURN
 * - SUCCESS | The newly created duplicated matrix
 * - ERROR   | NULL
 */
float** float_matrix_duplic(float** matrix, size_t height, size_t width)
{
  if(matrix == NULL || height <= 0 || width <= 0) return NULL;

  float** duplic = malloc(sizeof(float*) * height);

  if(duplic == NULL) return NULL;

  for(size_t index = 0; index < height; index++)
  {
    duplic[index] = float_vector_duplic(matrix[index], width);
  }
  return duplic;
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
