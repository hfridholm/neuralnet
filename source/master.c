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

  float inputs[4][2] = {{0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}};

  float targets[4][1] = {{0.0f}, {1.0f}, {0.0f}, {1.0f}};

  network_train_stcast_epochs(&network, (float**) inputs, (float**) targets, 4, 1);

  for(size_t index = 0; index < 4; index++)
  {
    float outputs[1];

    network_forward((float*) outputs, network, (float*) inputs[index]);

    info_print("[%.1f %.1f] => [%f]\n", inputs[index][0], inputs[index][1], outputs[0]);
  }

  network_free(&network);

  return 0;
}
