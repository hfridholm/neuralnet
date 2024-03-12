#ifndef MATRIX_H
#define MATRIX_H

#include <math.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>

// Float vector

extern float*   float_vector_create(size_t length);

extern void     float_vector_free(float** vector, size_t length);

extern float*   float_vector_copy(float* destin, const float* source, size_t length);

extern int      float_vector_minmax(float* min, float* max, const float* vector, size_t length);

extern float    float_random_create(float min, float max);

extern float*   float_vector_random_create(size_t length, float min, float max);

extern float*   float_vector_scale_multi(float* result, const float* vector, size_t length, float scalor);

extern float*   float_vector_elem_addit(float* result, const float* vector1, const float* vector2, size_t length);

extern float**  float_vector_dotprod(float** result, const float* vector1, size_t length1, const float* vector2, size_t length2);

extern void     float_vector_print(const float* vector, size_t length);

// Float matrix

extern float**  float_matrix_create(size_t height, size_t width);

extern void     float_matrix_free(float*** matrix, size_t height, size_t width);

extern float**  float_matrix_copy(float** destin, float** source, size_t height, size_t width);

extern float**  float_matrix_filter_index(float** result, float** matrix, size_t height, size_t width, const int* indexes, size_t amount);

extern float**  float_matrix_random_create(size_t height, size_t width, float min, float max);

extern float**  float_matrix_transp(float** transp, float** matrix, size_t height, size_t width);

extern float**  float_matrix_scale_multi(float** result, float** matrix, size_t height, size_t width, float scalor);

extern float**  float_matrix_elem_addit(float** result, float** matrix1, float** matrix2, size_t height, size_t width);

extern int      float_matrix_vector_dotprod(float* result, float** matrix, size_t height, size_t width, const float* vector);

extern void     float_matrix_print(float** matrix, size_t height, size_t width);

// Float matrix array

extern float*** float_matarr_create(size_t amount, size_t height, size_t width);

extern void     float_matarr_free(float**** matarr, size_t amount, size_t height, size_t width);

extern float*** float_matarr_scale_multi(float*** result, float*** matarr, size_t amount, size_t height, size_t width, float scalor);

extern float*** float_matarr_elem_addit(float*** result, float*** matarr1, float*** matarr2, size_t amount, size_t height, size_t width);

extern void     float_matarr_print(float*** matarr, size_t amount, size_t height, size_t width);

// Index array

extern size_t* index_array_shuffled_fill(size_t* array, size_t amount);

#endif // SECURE_H
