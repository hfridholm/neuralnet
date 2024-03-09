#include "review.h"
#include "persue.h"

#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

extern size_t network_max_layer_nodes(Network network);

int main(int argc, char* argv[])
{
  // srand(time(NULL));
  srand(420);

  info_print("Neural Network");

  char imgPath[] = "../assets/smilie.png";

  size_t imgWidth, imgHeight;
  float** matrix = image_values_matrix_read(&imgWidth, &imgHeight, imgPath);

  if(matrix == NULL)
  {
    error_print("Failed to read image\n");

    return 1;
  }


  float** inputs = float_matrix_create(imgWidth * imgHeight, 2);
  float** targets = float_matrix_create(imgWidth * imgHeight, 1);

  float_matrix_filter_index(inputs, matrix, imgWidth * imgHeight, 3, (int[]) {0, 1}, 2);
  float_matrix_filter_index(targets, matrix, imgWidth * imgHeight, 3, (int[]) {2}, 1);


  Network network;

  size_t amount = 7;
  size_t amounts[] = {2, 8, 8, 16, 8, 8, 1};
  activ_t activs[] = {ACTIV_RELU, ACTIV_RELU, ACTIV_RELU, ACTIV_RELU, ACTIV_RELU, ACTIV_SIGMOID};
  float learnrate = 0.01;
  float momentum = 0;

  int status = network_init(&network, amount, amounts, activs, learnrate, momentum);

  if(status != 0) error_print("network_init");

  network_print(network);


  network_train_stcast_epochs(&network, inputs, targets, imgWidth * imgHeight, 1);

  
  size_t outWidth = 256;
  size_t outHeight = 256;

  float outPixels[outWidth * outHeight];

  for(size_t yValue = 0; yValue < outHeight; yValue++)
  {
    for(size_t xValue = 0; xValue < outWidth; xValue++)
    {
      float outputs[1];

      float normX = (float) xValue / (outWidth - 1);
      float normY = (float) yValue / (outHeight - 1);

      float outInputs[2] = {normX, normY};

      network_forward(outputs, network, outInputs);

      size_t index = (yValue * outWidth + xValue);

      outPixels[index] = outputs[0];
    }
  }

  char outputPath[128] = "result.png";

  image_values_write(outputPath, outPixels, outWidth, outHeight);


  float_matrix_free(&inputs, imgWidth * imgHeight, 2);
  float_matrix_free(&targets, imgWidth * imgHeight, 1);

  float_matrix_free(&matrix, imgWidth * imgHeight, 3);

  network_free(&network);

  return 0;
}
