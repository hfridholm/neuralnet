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
 * Create a matrix array with random values
 *
 * RETURN
 * - SUCCESS | Matrix array with random values
 * - ERROR   | NULL
 */
float*** float_matarr_random_create(size_t amount, size_t height, size_t width, float min, float max)
{
  if(amount <= 0 || height <= 0 || width <= 0) return NULL;
 
  float*** matarr = malloc(sizeof(float**) * amount);

  if(matarr == NULL) return NULL;

  for(size_t index = 0; index < amount; index++)
  {
    matarr[index] = float_matrix_random_create(height, width, min, max);
  }
  return matarr;
}

/*
 * Fill a matrix array with random values
 *
 * RETURN
 * - SUCCESS | The filled matrix array with random values
 * - ERROR   | NULL
 */
float*** float_matarr_random_fill(float*** matarr, size_t amount, size_t height, size_t width, float min, float max)
{
  if(matarr == NULL) return NULL;

  for(size_t index = 0; index < amount; index++)
  {
    float_matrix_random_fill(matarr[index], height, width, min, max);
  }
  return matarr;
}

/*
 * Add the values of two matrix arrays together with each other
 *
 * RETURN
 * - SUCCESS | float*** result
 * - ERROR   | NULL
 */
float*** float_matarr_elem_addit(float*** result, float*** matarr1, float*** matarr2, size_t amount, size_t height, size_t width)
{
  if(result == NULL || matarr1 == NULL || matarr2 == NULL) return NULL;

  for(size_t index = 0; index < amount; index++)
  {
    float_matrix_elem_addit(result[index], matarr1[index], matarr2[index], height, width);
  }
  return result;
}

/*
 * Mulitply matrix array values by a scalor
 *
 * RETURN
 * - SUCCESS | float*** result
 * - ERROR   | NULL
 */
float*** float_matarr_scale_multi(float*** result, float*** matarr, size_t amount, size_t height, size_t width, float scalor)
{
  if(result == NULL || matarr == NULL) return NULL;

  for(size_t index = 0; index < amount; index++)
  {
    float_matrix_scale_multi(result[index], matarr[index], height, width, scalor);
  }
  return result;
}

/*
 * Copy the content of source to destin
 *
 * RETURN
 * - SUCCESS | float*** destin
 * - ERROR   | NULL
 */
float*** float_matarr_copy(float*** destin, float*** source, size_t amount, size_t height, size_t width)
{
  if(destin == NULL || source == NULL) return NULL;

  for(size_t index = 0; index < amount; index++)
  {
    float_matrix_copy(destin[index], source[index], height, width);
  }
  return destin;
}

/*
 * Print the inputted matrix to the console
 */
void float_matarr_print(float*** matarr, size_t amount, size_t height, size_t width)
{
  if(matarr == NULL) return;

  for(size_t index = 0; index < amount; index++)
  {
    float_matrix_print(matarr[index], height, width);

    printf("\n");
  }
}
