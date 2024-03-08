#include "../persue.h"

/*
 * Calculate the values of each node in the inputted network from the inputs
 */
int node_values_create(float** values, Network network, const float* inputs)
{
  if(values == NULL || inputs == NULL) return 1;

  float_vector_copy(values[0], inputs, network.inputs);

  size_t width = network.inputs;

  for(size_t index = 1; index < network.amount; index++)
  {
    NetworkLayer layer = network.layers[index - 1];

    float_matrix_vector_dotprod(values[index], layer.weights, layer.amount, width, values[index - 1]);

    float_vector_elements_add(values[index], values[index], layer.biases, layer.amount);

    activ_values(values[index], values[index], layer.amount, layer.activ);

    width = layer.amount;
  }
  return 0;
}

/*
 *
 */
int node_derivs_create(float** derivs, Network network, float** values, const float* targets)
{
  if(derivs == NULL || values == NULL || targets == NULL) return 1;

  // If no hidden or output layer exist, there is no need for derivatives
  if(network.amount <= 0) return 0; // Success!

  NetworkLayer outputLayer = network.layers[network.amount - 1];

  cross_entropy_derivs(derivs[network.amount - 1], values[network.amount], targets, outputLayer.amount);

  active_derivs_apply(derivs[network.amount - 1], values[network.amount], outputLayer.amount, outputLayer.activ);

  for(size_t index = (network.amount - 1); index >= 1; index--)
  {
    NetworkLayer layer = network.layers[index];

    size_t height = network.layers[index].amount;
    size_t width  = network.layers[index - 1].amount;

    float weightsTransp[layer.amount][height];
    float_matrix_transpose(weightsTransp, layer.weights, height, width)

    float_matrix_vector_dotprod(derivs[index - 1], weightsTransp, width, height, derivs[index]);

    activ_derivs_apply(derivs[index - 1], values[index], width, layer.activ);
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
int weight_bias_derivs_create(float*** wderivs, float** bderivs, Network network, const float* inputs, const float* targets)
{
  if(wderivs == NULL || bderivs == NULL || inputs == NULL || targets == NULL) return 1;

  float nvalues[network.amount + 1][maxSize];
  float nderivs[network.amount][maxSize];

  node_values_create(nvalues, network, inputs);
  node_derivs_create(nderivs, network, nvalues, targets);

  for(size_t index = (network.amount - 1); index >= 0; index--)
  {
    NetworkLayer layer = network.layers[index];

    size_t height = network.layers[index].amount;
    size_t width  = network.layers[index - 1].amount;

    float_vector_dotprod(wderivs[index], nderivs[index], height, nvalues[index], width);

    float_vector_copy(bderivs[index], nderivs[index], height);
  }
  return 0; // Success!
}

static int layer_weight_deltas_create(float** wdeltas, float** twdeltas, float** wderivs, size_t height, size_t width, float learnrate, float momentum)
{
  float_matrix_scale_mutli(twdeltas, wderivs, height, width, -learnrate);

  if(wdeltas != NULL)
  {
    float_matrix_scale_mutli(twdeltas, wdeltas, height, width, momentum);

    float_matrix_elem_addit(wdeltas, wdeltas, twdeltas, height, width);
  }
}

static int layer_bias_deltas_create(float* bdeltas, float* tbdeltas, float* bderivs, size_t height, float learnrate, float momentum)
{
  float_vector_scale_mutli(tbdeltas, bderivs, height, -learnrate);

  if(bdeltas != NULL)
  {
    float_vector_scale_mutli(tbdeltas, bdeltas, height, momentum);

    float_vector_elem_addit(bdeltas, bdeltas, tbdeltas, height);
  }
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
int weight_bias_deltas_create(float*** wdeltas, float** bdeltas, Network network, const float* inputs, const float* targets)
{
  if(wdeltas == NULL || bdeltas == NULL || inputs == NULL || targets == NULL) return 1;

  size_t maxSize = network_max_layer_nodes(network);

  float twdeltas[network.amount][maxSize][maxSize]; // Temporary weight delta values
  float tbdeltas[network.amount][maxSize]; // Temporary bias delta values

  float wderivs[network.amount][maxSize][maxSize]; // Weight derivatives
  float bderivs[network.amount][maxSize]; // Bias derivatives

  weight_bias_derivs_create(wderivs, bderivs, network, inputs, targets);

  size_t width = network.inputs;

  for(size_t index = 0; index < network.amount; index++)
  {
    NetworkLayer layer = network.layers[index];

    size_t height = layer.amount;

    layer_weight_deltas_create(wdeltas[index], twdelats[index], wderivs[index], height, width, network.learnrate, network.momentum);

    layer_bias_deltas_create(bdeltas[index], tbdelats[index], bderivs[index], height, network.learnrate, network.momentum);

    // The width of the next layer is the height of the current layer
    width = layer.amount;
  }
  return 0; // Success!
}

/*
 * Train the network stochastically on a single sample
 *
 * RETURN (int status)
 * - 0 | Success!
 * - 1 | The inputted arguments are bad
 */
static int network_train_stcast_sample(Network network, const float* inputs, const float* targets)
{
  if(inputs == NULL || targets == NULL) return 1;

  weight_bias_deltas_create(wdeltas, bdeltas, network, inputs, targets);

  size_t width = network.inputs;

  for(size_t index = 0; index < network.amount; index++)
  {
    NetworkLayer layer = network.layers[index];

    size_t height = layer.amount;

    float_matrix_elem_addit(layer.weights, layer.weights, wdeltas, height, width);
    
    float_vector_elem_addit(layer.biases, layer.biases, bdeltas, height);

    width = layer.amount;
  }
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

  for(size_t index = 0; index < epochs; index++)
  {
    int status = network_train_stcast_epoch(network, inputs, targets, amount);

    if(status != 0) return 2;
  }
  return 0;
}
