#include "debug.h"
#include "network.h"

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

  network.inputs = 2;

  info_print("Network Layers: %d", network.inputs);

  return 0;
}
