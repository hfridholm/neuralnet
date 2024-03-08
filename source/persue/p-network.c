#include "../persue.h"
#include "p-activs-intern.h"

size_t network_max_layer_nodes(Network network)
{
  size_t maxSize = network.inputs;

  for(size_t index = 0; index < network.amount; index++)
  {
    size_t currSize = network.layers[index].amount;

    if(currSize > maxSize) maxSize = currSize;
  }
  return maxSize;
}

int network_forward(float* outputs, Network network, const float* inputs)
{
  if(outputs == NULL || inputs == NULL) return 1;

  size_t maxSize = network_max_layer_nodes(network);

  float tempOutputs[maxSize];

  size_t width = network.inputs;

  for(size_t index = 0; index < network.amount; index++)
  {
    NetworkLayer layer = network.layers[index];

    float_matrix_vector_dotprod(tempOutputs, layer.weights, layer.amount, width, tempOutputs);

    float_vector_elements_add(tempOutputs, tempOutputs, layer.biases, layer.amount);

    activ_values(tempOutputs, tempOutputs, layer.amount, layer.activ);

    // The width of the next layer is the height of this layer
    width = layer.amount;
  }
  // Width is the size of the last layer (output layer)
  outputs = float_vector_copy(outputs, tempOutputs, width);

  return 0; // Success
}

/*
 * Initialize the values of a NetworkLayer struct
 *
 * Note: This function is designed for performence, not safety
 *
 * RETURN (int status)
 * - 0 | Success!
 * - 1 | Inputted arguments are bad
 */
int network_layer_init(NetworkLayer* layer, size_t amount, size_t inputs, activ_t activ)
{
  // If the inputted arguments are bad
  if(layer == NULL || amount <= 0 || inputs <= 0) return 1;

  layer->weights = float_matrix_create(amount, inputs);
  layer->biases = float_vector_create(amount);

  layer->amount = amount;
  layer->activ = activ;

  return 0; // Success!
}

/*
 * Initialize the values of a Network struct
 *
 * PARAMS
 * - Network* network      | The pointer to the Network struct
 * - size_t amount         | The amount of layers (input, hiddens, output)
 * - const size_t* amounts | The sizes of each layer. The amount of nodes in each layer
 *   Size: amount
 * - const activ_t* activs | The activation function for each layer (ex input)
 *   Size: amount - 1 (ex input layer)
 *
 * RETURN (int status)
 * - 0 | Success!
 * - 1 | The inputted arguments are bad
 */
int network_init(Network* network, size_t amount, const size_t* amounts, const activ_t* activs, float learnrate, float momentum)
{
  // If the inputted arguments are bad
  if(network == NULL || amount <= 0 || amounts == NULL || activs == NULL) return 1;

  network->inputs = amounts[0];
  network->amount = (amount - 1);

  network->layers = malloc(sizeof(NetworkLayer) * (amount - 1));
  
  for(size_t index = 0; index < (amount - 1); index++)
  {
    int status = network_layer_init(&network->layers[index], amounts[index + 1], amounts[index], activs[index]);

    // If the current layer failed to be initialized
    if(status != 0)
    {
      error_print("Failed to initialize network layer");

      network_free(network);

      return 2;
    }
  }
  network->learnrate = learnrate;
  network->momentum = momentum;

  return 0; // Success!
}

/*
 * Free the allocated memory in the inputted NetworkLayer struct
 *
 * PARAMS
 * - NetworkLayer* layer | A pointer to the NetworkLayer struct
 * - size_t inputs       | The amount of ingoing nodes to the layer
 */
void network_layer_free(NetworkLayer* layer, size_t inputs)
{
  float_matrix_free(&layer->weights, layer->amount, inputs);

  float_vector_free(&layer->biases, layer->amount);
}

/*
 * Free the allocated memory in the inputted Network struct
 *
 * PARAMS
 * - Network* network | A pointer to the Network struct
 */
void network_free(Network* network)
{
  // If the layers are not allocated, there is nothing to free
  if(network->layers == NULL) return;

  size_t inputs = network->inputs;

  for(size_t index = 0; index < network->amount; index++)
  {
    // Moving the pointer to the next layer
    NetworkLayer* layer = (network->layers + index);
  
    network_layer_free(layer, inputs);

    inputs = layer->amount;
  }
  free(network->layers);

  network->layers = NULL;
}
