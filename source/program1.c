#include "review.h"
#include "persue.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[])
{
  // srand(time(NULL));
  srand(420);

  error_print("Neural Network");

  Network network;

  size_t amount = 3;
  size_t amounts[] = {2, 4, 1};
  activ_t activs[] = {ACTIV_SIGMOID, ACTIV_SIGMOID};
  float learnrate = 0.01;
  float momentum = 0.01;

  network_init(&network, amount, amounts, activs, learnrate, momentum);

  float tinputs[4][2] = {{1.0f, 1.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}, {0.0f, 0.0f}};
  float ttargets[4][1] = {{0.0f}, {0.0f}, {0.0f}, {1.0f}};

  float** inputs = float_matrix_create(4, 2);
  float** targets = float_matrix_create(4, 1);

  for(size_t index = 0; index < 4; index++)
  {
    inputs[index][0] = tinputs[index][0];
    inputs[index][1] = tinputs[index][1];

    targets[index][0] = ttargets[index][0];
  }

  for(size_t index = 0; index < 4; index++)
  {
    float outputs[1];

    network_forward(outputs, network, inputs[index]);

    printf("[%.1f %.1f] => [%f] (%.1f)\n", inputs[index][0], inputs[index][1], outputs[0], targets[index][0]);
  }

  size_t outputAmount = network.layers[network.amount - 1].amount;

  float toutputs[outputAmount];

  network_forward(toutputs, network, inputs[0]);
  printf("===== WEIGHTS BEFORE =====\n");
  float_matrix_print(network.layers[0].weights, network.layers[0].amount, network.inputs);
  float_matrix_print(network.layers[1].weights, network.layers[1].amount, network.layers[0].amount);
  printf("===== WEIGHTS BEFORE END =====\n");
  float_vector_print(network.layers[0].biases, network.layers[0].amount);
  float_vector_print(network.layers[1].biases, network.layers[1].amount);
  printf("===== BIAS BEFORE END ========\n");
  printf("Cost: %.2f\n", cross_entropy_cost(toutputs, targets[0], outputAmount));

  network_train_stcast_epochs(&network, inputs, targets, 1, 10);

  network_forward(toutputs, network, inputs[0]);
  printf("===== WEIGHTS AFTER =====\n");
  float_matrix_print(network.layers[0].weights, network.layers[0].amount, network.inputs);
  float_matrix_print(network.layers[1].weights, network.layers[1].amount, network.layers[0].amount);
  printf("===== WEIGHTS AFTER END =====\n");
  float_vector_print(network.layers[0].biases, network.layers[0].amount);
  float_vector_print(network.layers[1].biases, network.layers[1].amount);
  printf("===== BIAS AFTER END ========\n");
  printf("Cost: %.2f\n", cross_entropy_cost(toutputs, targets[0], outputAmount));

  for(size_t index = 0; index < 4; index++)
  {
    float outputs[1];

    network_forward(outputs, network, inputs[index]);

    printf("[%.1f %.1f] => [%f] (%.1f)\n", inputs[index][0], inputs[index][1], outputs[0], targets[index][0]);
  }

  float_matrix_free(&inputs, 4, 2);
  float_matrix_free(&targets, 4, 1);

  network_free(&network);

  return 0;
}
