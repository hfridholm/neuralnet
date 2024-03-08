#include "../persue.h"

static int image_pixels_write(const char* filepath, const uint8_t* pixels, size_t width, size_t height)
{
  if(!stbi_write_png(filepath, width, height, 1, pixels, width * sizeof(uint8_t)))
  {
    error_print("stbi_write_png\n");
  
    return 1;
  }
  return 0; // Success!
}

int image_values_write(const char* filepath, const float* values, size_t width, size_t height)
{
  uint8_t pixels[width * height];

  for(size_t index = 0; index < (width * height); index++)
  {
    pixels[index] = (uint8_t) (values[index] * 255); 
  }

  int status = image_pixels_write(filepath, pixels, width, height);

  return (status == 0) ? 0 : 1;
}

static uint8_t* image_pixels_read(size_t* width, size_t* height, const char* filepath)
{
  int twidth, theight, tcomp;

  uint8_t* pixels = (uint8_t*) stbi_load(filepath, &twidth, &theight, &tcomp, 0);

  if(pixels == NULL)
  {
    error_print("stbi_load: %s", strerror(errno));

    return NULL;
  }

  if(tcomp != 1)
  {
    free(pixels);

    error_print("Image comp != 1");

    return NULL;
  }

  *width = twidth;
  *height = theight;

  return pixels;
}

float* image_values_read(size_t* width, size_t* height, const char* filepath)
{
  uint8_t* pixels = image_pixels_read(width, height, filepath);

  if(pixels == NULL) return NULL;
  
  size_t length = (*width * *height);

  float* values = float_vector_create(length);

  for(size_t index = 0; index < length; index++)
  {
    values[index] = (float) pixels[index] / 255.f;
  }
  free(pixels);

  return values;
}

float** image_values_matrix_read(size_t* width, size_t* height, const char* filepath)
{
  uint8_t* pixels = image_pixels_read(width, height, filepath);

  if(pixels == NULL) return NULL;
  
  float** matrix = float_matrix_create(*width * *height, 3);

  for(size_t yValue = 0; yValue < *height; yValue++)
  {
    for(size_t xValue = 0; xValue < *width; xValue++)
    {
      size_t index = (yValue * *width + xValue);

      matrix[index][0] = (float) xValue / (*width - 1);
      matrix[index][1] = (float) yValue / (*height - 1);
      matrix[index][2] = (float) pixels[index] / 255.0f;
    }
  }
  free(pixels);

  return matrix;
}
