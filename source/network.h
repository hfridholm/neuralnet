#ifndef NETWORK_H
#define NETWORK_H

#include "debug.h"

#include <stddef.h>
#include <stdlib.h>

// This are identifiers for different activation functions
typedef enum { ACTIV_NONE, ACTIVE_SIGMOID, ACTIVE_RELU, ACTIVE_TANH, ACTIVE_SOFTMAX } activ_t;

typedef struct
{
  size_t amount;   // The amount of nodes
  // The size of the weights matrix:
  // the amount of nodes in this layer x the amount of nodes in the previous layer
  float** weights; // The weights for this layers nodes
  float* biases;   // The biases for this layers nodes
  activ_t activ;   // The activation function identifier
} NetworkLayer;

typedef struct
{
  size_t inputs;        // The amount of input nodes (values)
  size_t amount;        // The amount of layers
  NetworkLayer* layers; // The hidden layers and the output layer
  float learnrate;      // The learning rate
  float momentum;       // The momentum
} Network;

extern int network_init(Network* network, size_t amount, const size_t* amounts, const activ_t* activs, float learnrate, float momentum);

extern void network_free(Network* network);

#endif // NETWORK_H
