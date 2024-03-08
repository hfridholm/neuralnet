#include "review.h"
#include "persue.h"

#include <stdio.h>
#include <stdlib.h>

/*
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
*/

int main(int argc, char* argv[])
{
  error_print("Neural Network");

  Network network;

  size_t amount = 3;
  size_t amounts[] = {2, 3, 1};
  activ_t activs[] = {ACTIV_NONE, ACTIV_SIGMOID};
  float learnrate = 0.03;
  float momentum = 0.2;

  network_init(&network, amount, amounts, activs, learnrate, momentum);

  network_free(&network);

  return 0;
}
