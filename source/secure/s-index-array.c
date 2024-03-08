#include "../secure.h"

/*
 * Return random index between min and max
 */
static size_t index_random_create(size_t min, size_t max)
{
  float fraction = ((float) rand() / (float) RAND_MAX);

  return (fraction * (max - min) + min);
}

/*
 * Swap two indexes in index array
 */
static size_t* index_array_switch_index(size_t* array, size_t index1, size_t index2)
{
  size_t temp = array[index1];

  array[index1] = array[index2];
  array[index2] = temp;

  return array; 
}

/*
 * Fill the inputted array with shuffled indexes
 * Each index only appears once
 *
 * RETURN (size_t* array)
 * - SUCCESS | size_t* array
 * - ERROR   | NULL
 */
size_t* index_array_shuffled_fill(size_t* array, size_t amount)
{
  if(array == NULL) return NULL;

  for(size_t index = 0; index < amount; index++)
  {
    array[index] = index;
  }
  for(size_t index = 0; index < amount; index++)
  {
    size_t random = index_random_create(0, amount - 1);

    array = index_array_switch_index(array, index, random);
  }
  return array;
}
