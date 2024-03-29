#ifndef PERSUE_H
#define PERSUE_H

#include "review.h"
#include "secure.h"

#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>

// This are identifiers for different activation functions
typedef enum { ACTIV_NONE, ACTIV_SIGMOID, ACTIV_RELU, ACTIV_TANH, ACTIV_SOFTMAX } activ_t;

typedef struct
{
  size_t amount;   // The amount of nodes
  // The size of the weights matrix:
  // the amount of nodes in this layer x the amount of nodes in the previous layer
  float** weights; // The weights for this layers nodes
  float* biases;   // The biases for this layers nodes
  activ_t activ;   // The activation function identifier
  // This data is keept for use of the momentum
  float** wdeltas; // The delta values of the weight derivatives
  float* bdeltas;  // The delta values of the bias derivatives
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

extern void network_print(Network network);

extern int network_forward(float* outputs, Network network, const float* inputs);

extern int network_train_stcast_epochs(Network* network, float** inputs, float** targets, size_t amount, size_t epochs);

extern int network_train_stcast(Network* network, const float* inputs, const float* targets);

extern int network_train_mini_batch(Network* network, float** inputs, float** targets, size_t amount);

extern int network_train_mini_batch_epochs(Network* network, float** inputs, float** targets, size_t amount, size_t bsize, size_t epochs);

extern float cross_entropy_cost(const float* nodes, const float* targets, size_t amount);

#endif // PERSUE_H
