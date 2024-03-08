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

// Float Vector

extern float*   float_vector_create(size_t length);

extern void     float_vector_free(float** vector, size_t length);


extern float*   float_vector_copy(float* destin, const float* source, size_t length);

extern float*   float_vector_duplic(const float* vector, size_t length);


extern int      float_vector_minmax(float* min, float* max, const float* vector, size_t length);

extern float    float_vector_sum(const float* vector, size_t length);

// Random
extern float    float_random_create(float min, float max);

extern float*   float_vector_random_create(size_t length, float min, float max);

extern float*   float_vector_random_fill(float* vector, size_t length, float min, float max);


extern float*   float_vector_scale_multi(float* result, const float* vector, size_t length, float scalor);

extern float*   float_vector_scale_addit(float* result, const float* vector, size_t length, float scalor);


extern float*   float_vector_elem_multi(float* result, const float* vector1, const float* vector2, size_t length);

extern float*   float_vector_elem_addit(float* result, const float* vector1, const float* vector2, size_t length);


extern float**  float_vector_dotprod(float** result, const float* vector1, size_t length1, const float* vector2, size_t length2);


extern void     float_vector_print(const float* vector, size_t length);

// Float Matrix

extern float**  float_matrix_create(size_t height, size_t width);

extern void     float_matrix_free(float*** matrix, size_t height, size_t width);

// Copy
extern float**  float_matrix_copy(float** destin, float** source, size_t height, size_t width);

extern float**  float_matrix_duplic(float** matrix, size_t height, size_t width);

// Filter
extern float**  float_matrix_filter_index(float** result, float** matrix, size_t height, size_t width, const int* indexes, size_t amount);

// Random
extern float**  float_matrix_random_create(size_t height, size_t width, float min, float max);

extern float**  float_matrix_random_fill(float** matrix, size_t height, size_t width, float min, float max);


extern float**  float_matrix_transp(float** transp, float** matrix, size_t height, size_t width);

// Scale
extern float**  float_matrix_scale_multi(float** result, float** matrix, size_t height, size_t width, float scalor);

extern float**  float_matrix_scale_addit(float** result, float** matrix, size_t height, size_t width, float scalor);

// Elem
extern float**  float_matrix_elem_multi(float** result, float** matrix1, float** matrix2, size_t height, size_t width);

extern float**  float_matrix_elem_addit(float** result, float** matrix1, float** matrix2, size_t height, size_t width);

// Vector
extern float*   float_matrix_column_vector(float* vector, float** matrix, size_t height, size_t width, size_t column);

extern float**  float_matrix_vector_addit(float** result, float** matrix, const float* vector, size_t height, size_t width);

// Dotprod
extern int      float_matrix_vector_dotprod(float* result, float** matrix, size_t height, size_t width, const float* vector);

// Convert
extern float**  float_vector_cnvrt_matrix(float** matrix, const float* vector, size_t length);

extern float*   float_matrix_cnvrt_vector(float* vector, float** matrix, size_t height, size_t width);

// Print
extern void     float_matrix_print(float** matrix, size_t height, size_t width);


extern float*** float_matarr_create(size_t amount, size_t height, size_t width);

extern void float_matarr_free(float**** matarr, size_t amount, size_t height, size_t width);

// Index Array

extern size_t* index_array_shuffled_fill(size_t* array, size_t amount);

#endif // SECURE_H
