#ifndef WONDER_H
#define WONDER_H

#include "secure.h"
#include "review.h"

#include <errno.h>

#include "stb_image.h"
#include "stb_image_write.h"

extern int image_values_write(const char* filepath, const float* values, size_t width, size_t height);

extern float** image_values_matrix_read(size_t* width, size_t* height, const char* filepath);

#endif // WONDER_H
