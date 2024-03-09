#include "../persue.h"
#include "../review.h"

#include "p-activs-intern.h"
#include "p-network-intern.h"

/*
 * Calculate the values of each node in the inputted network from the inputs
 */
static int node_values_create(float** values, Network network, const float* inputs)
{
  if(values == NULL || inputs == NULL) return 1;

  float_vector_copy(values[0], inputs, network.inputs);

  size_t width = network.inputs;

  // From the first hidden layer (second layer) to the last layer
  for(size_t index = 1; index <= network.amount; index++)
  {
    NetworkLayer layer = network.layers[index - 1];

    size_t height = layer.amount;

    float_matrix_vector_dotprod(values[index], layer.weights, height, width, values[index - 1]);

    float_vector_elem_addit(values[index], values[index], layer.biases, height);

    activ_values(values[index], values[index], height, layer.activ);

    width = layer.amount;
  }
  return 0;
}

/*
 *
 */
static int node_derivs_create(float** derivs, Network network, float** values, const float* targets)
{
  if(derivs == NULL || values == NULL || targets == NULL) return 1;

  // If no hidden or output layer exist, there is no need for derivatives
  if(network.amount <= 0) return 0; // Success!

  // 1. Calculating the derivatives for the output layer
  NetworkLayer outputLayer = network.layers[network.amount - 1];

  cross_entropy_derivs(derivs[network.amount - 1], values[network.amount], targets, outputLayer.amount);

  activ_derivs_apply(derivs[network.amount - 1], values[network.amount], outputLayer.amount, outputLayer.activ);

  // 2. Calculating the derivatives for the hidden layers
  // From the last hidden layer (next to last layer) to the first hidden layer (first layer)
  for(size_t index = (network.amount - 1); index-- >= 1;)
  {
    NetworkLayer layer = network.layers[index];

    // The height and with will be the height and with for the layer before (closer to output)
    size_t height = network.layers[index + 1].amount;
    size_t width = network.layers[index].amount;
    // Result: width is the height of the current layer

    // The weights are from the layer before (close to output)
    float** weights = network.layers[index + 1].weights;
    float** weightsTransp = float_matrix_create(width, height);

    float_matrix_transp(weightsTransp, weights, height, width);

    // derivs[index + 1] is the derivs from the layer before (closer to output layer)
    float_matrix_vector_dotprod(derivs[index], weightsTransp, width, height, derivs[index + 1]);

    float_matrix_free(&weightsTransp, width, height);

    activ_derivs_apply(derivs[index], values[index + 1], width, layer.activ);
  }
  return 0; // Success!
}

/*
 * PARAMS
 * - float*** wderivs     | The derivatives for the weights
 * - float** bderivs      | The derivatives for the biases
 * - Network network      | The nerual network
 * - const float* inputs  | The inputs
 * - const float* targets | The targets
 */
static int weight_bias_derivs_create(float*** wderivs, float** bderivs, Network network, const float* inputs, const float* targets)
{
  if(wderivs == NULL || bderivs == NULL || inputs == NULL || targets == NULL) return 1;

  size_t maxSize = network_max_layer_nodes(network);

  float** nvalues = float_matrix_create(network.amount + 1, maxSize);
  float** nderivs = float_matrix_create(network.amount, maxSize);

  node_values_create(nvalues, network, inputs);
  node_derivs_create(nderivs, network, nvalues, targets);

  // From the last layer (output layer) to the first layer (first hidden layer)
  for(size_t index = network.amount; index-- >= 1;)
  {
    size_t height = network.layers[index].amount;

    // If the layer is the first hidden layer, then the width is the amount of input nodes,
    // else, the width is the amount of nodes in the layer before
    size_t width = (index >= 1) ? network.layers[index - 1].amount : network.inputs;

    float_vector_dotprod(wderivs[index], nderivs[index], height, nvalues[index], width);

    float_vector_copy(bderivs[index], nderivs[index], height);
  }
  float_matrix_free(&nvalues, network.amount + 1, maxSize);
  float_matrix_free(&nderivs, network.amount, maxSize);

  return 0; // Success!
}

static int layer_weight_deltas_create(float** wdeltas, float** wderivs, size_t height, size_t width, float learnrate, float momentum)
{
  float** twdeltas = float_matrix_create(height, width);

  float_matrix_scale_multi(twdeltas, wderivs, height, width, -learnrate);

  // If old weight deltas exist, add a small part of the old deltas to the new deltas
  // This keeps the "momentum" going
  // Note: If the old deltas don't exist (the values are 0), then no this will have no effect
  if(wdeltas != NULL)
  {
    // Old weight deltas (wdeltas) x momentum + new weight deltas (twdeltas)
    float_matrix_scale_multi(wdeltas, wdeltas, height, width, momentum);

    float_matrix_elem_addit(wdeltas, wdeltas, twdeltas, height, width);
  }
  else float_matrix_copy(wdeltas, twdeltas, height, width);

  float_matrix_free(&twdeltas, height, width);

  return 0; // Success!
}

static int layer_bias_deltas_create(float* bdeltas, float* bderivs, size_t height, float learnrate, float momentum)
{
  float tbdeltas[height]; // Temporary bias delta values

  float_vector_scale_multi(tbdeltas, bderivs, height, -learnrate);

  // If old weight deltas exist, add a small part of the old deltas to the new deltas
  // This keeps the "momentum" going
  // Note: If the old deltas don't exist (the values are 0), then no this will have no effect
  if(bdeltas != NULL)
  {
    // Old weight deltas (bdeltas) x momentum + new weight deltas (tbdeltas)
    float_vector_scale_multi(bdeltas, bdeltas, height, momentum);

    float_vector_elem_addit(bdeltas, bdeltas, tbdeltas, height);
  }
  else float_vector_copy(tbdeltas, bdeltas, height);

  return 0; // Success!
}

/*
 * Create the deltas for both the weights and the biases for every layer (hiddens, output)
 *
 * PARAMS
 * - float*** wdeltas     | The weight deltas (optionally the old weight deltas)
 * - float** bdeltas      | The bias deltas (optionally the old bias deltas)
 * - Network network      | The neural network
 * - const float* inputs  | The inputs
 * - const float* targets | The targets
 *
 * RETURN (int status)
 * - 0 | Success!
 * - 1 | The inputted arguements are bad
 */
static int weight_bias_deltas_create(Network* network, const float* inputs, const float* targets)
{
  if(inputs == NULL || targets == NULL) return 1;

  size_t maxSize = network_max_layer_nodes(*network);

  float*** wderivs = float_matarr_create(network->amount, maxSize, maxSize); // Weight derivatives
  float** bderivs  = float_matrix_create(network->amount, maxSize);          // Bias derivatives

  int status = weight_bias_derivs_create(wderivs, bderivs, *network, inputs, targets);

  if(status != 0) error_print("weight_bias_derivs_create");

  size_t width = network->inputs;

  for(size_t index = 0; index < network->amount; index++)
  {
    NetworkLayer* layer = &network->layers[index];

    size_t height = layer->amount;

    layer_weight_deltas_create(layer->wdeltas, wderivs[index], height, width, network->learnrate, network->momentum);

    layer_bias_deltas_create(layer->bdeltas, bderivs[index], height, network->learnrate, network->momentum);

    // The width of the next layer is the height of the current layer
    width = layer->amount;
  }
  float_matarr_free(&wderivs, network->amount, maxSize, maxSize);
  float_matrix_free(&bderivs, network->amount, maxSize);

  return 0; // Success!
}

float cost = 0;

/*
 * Train the network stochastically on a single sample
 *
 * RETURN (int status)
 * - 0 | Success!
 * - 1 | The inputted arguments are bad
 */
int network_train_stcast_sample(Network* network, const float* inputs, const float* targets)
{
  if(inputs == NULL || targets == NULL) return 1;

  int status = weight_bias_deltas_create(network, inputs, targets);
  
  if(status != 0) error_print("weight_bias_deltas_create");

  size_t width = network->inputs;

  for(size_t index = 0; index < network->amount; index++)
  {
    NetworkLayer* layer = &network->layers[index];

    size_t height = layer->amount;

    float_matrix_elem_addit(layer->weights, layer->weights, layer->wdeltas, height, width);
    
    float_vector_elem_addit(layer->biases, layer->biases, layer->bdeltas, height);

    width = layer->amount;
  }

  float outputs[1];

  network_forward(outputs, *network, inputs);

  cost += cross_entropy_cost(outputs, targets, 1);

  return 0;
}

/*
 * Train the network stochastically on an epoch
 *
 * PARAMS
 * - Network* network | The neural network
 * Size: epochs x inputs
 * - float** inputs   | An array of inputs 
 * Size: epochs x outputs
 * - float** targets  | An array of targets
 * - size_t amount    | The size of the epoch / the batch size
 *
 * RETURN (int status)
 * - 0 | Success!
 * - 1 | Something else went wrong
 */
static int network_train_stcast_epoch(Network* network, float** inputs, float** targets, size_t amount)
{
  // No need to check input paramters,
  // because this function is not going to be called by a user

  size_t randomIndexes[amount];
  index_array_shuffled_fill(randomIndexes, amount);

  for(size_t index = 0; index < amount; index++)
  {
    size_t randomIndex = randomIndexes[index];

    int status = network_train_stcast_sample(network, inputs[randomIndex], targets[randomIndex]);

    if(status != 0) return 1;
  }
  return 0; // Success!
}

/*
 * Train the network stochastically multiple epochs
 *
 * PARAMS
 * - Network* network | The neural network
 * Size: epochs x inputs
 * - float** inputs   | An array of inputs 
 * Size: epochs x outputs
 * - float** targets  | An array of targets
 * - size_t amount    | The size of the epoch
 * - size_t epochs    | The amount of epochs
 *
 * RETURN (int status)
 * - 0 | Success!
 * - 1 | The inputted arguments are bad
 * - 2 | Something else went wrong
 */
int network_train_stcast_epochs(Network* network, float** inputs, float** targets, size_t amount, size_t epochs)
{
  if(network == NULL || inputs == NULL || targets == NULL) return 1;

  info_print("Training stochastically %d epochs", epochs);

  for(size_t index = 0; index < epochs; index++)
  {
    // info_print("Training epoch #%d with %d samples", index + 1, amount);

    int status = network_train_stcast_epoch(network, inputs, targets, amount);

    if(status != 0) return 2;

    printf("Mean Cost #%02ld: %f\n", index + 1, cost / amount);

    cost = 0;
  }
  return 0;
}
