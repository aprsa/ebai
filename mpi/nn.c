#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "nn.h"

#define SUCCESS 0

#define VERBOSE 0

#define THETA 0.75
#define TAU   0.35

NN_layer *nn_layer_new (int nodes_no)
{
	NN_layer *layer = malloc (sizeof (*layer));
	layer->stimulus = malloc (nodes_no * sizeof (*(layer->stimulus)));
	layer->nodes_no = nodes_no;
	layer->prev     = NULL;
	layer->next     = NULL;

	return layer;
}

int nn_layer_zero_stimulus (NN_layer *layer)
{
	int i;

	for (i = 0; i < layer->nodes_no; i++)
		layer->stimulus[i] = 0.0;

	return SUCCESS;
}

int nn_layer_stimulate (NN_layer *layer, NN_transfer_function activation, double *stimulus)
{
	int i;

	for (i = 0; i < layer->nodes_no; i++)
		switch (activation) {
			case NN_TF_LINEAR:
				layer->stimulus[i] = stimulus[i];
			break;
			case NN_TF_SIGMOID:
				layer->stimulus[i] = nn_tf_sigmoid (stimulus[i], THETA, TAU);
			break;
			default:
				return NN_ERROR_TF_INVALID;
		}

	return SUCCESS;
}

int nn_layer_propagate (NN_layer *layer, NN_connection *connection, NN_direction direction, NN_transfer_function output_function)
{
	/*
	 * This function propagates the values from the passed layer to the
	 * subsequent layer in the following way:
	 *
	 *   net_i -> Af (net_i) -> Of (Af (net_i))
	 *
	 * where
	 *
	 *   net_i  ..  weighted sum of all input values
	 *   Af     ..  activation function (assumed linear)
	 *   Of     ..  output function
	 *
	 * The output of Of (Af (net_i)) is stored as the node value.
	 */

	int i, j;

	if (!layer) return NN_ERROR_LAYER_INVALID;
	if (!layer->next) return NN_ERROR_LAYER_NEXT_INVALID;

	if (direction == NN_FORWARD) {
		nn_layer_zero_stimulus (layer->next);
		for (i = 0; i < layer->next->nodes_no; i++) {
			for (j = 0; j < layer->nodes_no; j++)
				layer->next->stimulus[i] += connection->weights[i][j] * layer->stimulus[j];

			/* Here we assumed that the activation function is linear. */

			if (output_function == NN_TF_LINEAR)
				layer->next->stimulus[i] = nn_tf_linear (layer->next->stimulus[i]);
			if (output_function == NN_TF_SIGMOID)
				layer->next->stimulus[i] = nn_tf_sigmoid (layer->next->stimulus[i], THETA, TAU);
		}
	}

	return SUCCESS;
}

int nn_layer_print (NN_layer *layer)
{
	int i;

	for (i = 0; i < layer->nodes_no; i++)
		printf ("%lf\t", layer->stimulus[i]);
	printf ("\n");

	return SUCCESS;
}

int nn_layer_free (NN_layer *layer)
{
	free (layer->stimulus);
	free (layer);

	return SUCCESS;
}

NN_connection *nn_connection_new (NN_layer *from, NN_layer *to)
{
	int i;

	NN_connection *connection = malloc (sizeof (*connection));
	connection->from    = from;
	connection->to      = to;

	connection->weights = malloc (to->nodes_no * sizeof (*(connection->weights)));
	for (i = 0; i < to->nodes_no; i++)
		connection->weights[i] = malloc (from->nodes_no * sizeof (**(connection->weights)));

	from->next = to;
	to->prev = from;

	return connection;
}

int nn_connection_randomize (NN_connection *connection)
{
	int i, j;

	for (i = 0; i < connection->to->nodes_no; i++)
		for (j = 0; j < connection->from->nodes_no; j++)
			connection->weights[i][j] = -0.5 + (double) rand () / RAND_MAX;

	return SUCCESS;
}

int nn_connection_print (NN_connection *connection)
{
	int i, j;

	for (i = 0; i < connection->to->nodes_no; i++) {
		printf ("\t");
		for (j = 0; j < connection->from->nodes_no; j++)
			printf ("%lf\t", connection->weights[i][j]);
		printf ("\n");
	}

	return SUCCESS;
}

int nn_connection_print_to_file (char *filename, NN_connection *connection)
{
	int i, j;
	FILE *out = fopen (filename, "w");

	if (!out) {
		printf ("*** error: cannot create file %s, aborting.\n", filename);
		return NN_ERROR_FILENAME_INVALID;
	}

	for (i = 0; i < connection->to->nodes_no; i++) {
		for (j = 0; j < connection->from->nodes_no; j++)
			fprintf (out, "%lf\t", connection->weights[i][j]);
		fprintf (out, "\n");
	}

	fclose (out);
	return SUCCESS;
}

int nn_connection_free (NN_connection *connection)
{
	int i;

	for (i = 0; i < connection->to->nodes_no; i++)
		free (connection->weights[i]);
	free (connection->weights);
	free (connection);

	return SUCCESS;
}

double nn_tf_linear (double in)
{
	return in;
}

double nn_tf_sigmoid (double in, double theta, double tau)
{
	return 1.0/(1.0+exp(-(in-theta)/tau));
}

NN_exemplar *nn_exemplar_new (int inodes_no, int onodes_no)
{
	NN_exemplar *exemplar = malloc (sizeof (*exemplar));

	exemplar->inodes_no = inodes_no;
	exemplar->onodes_no = onodes_no;
	exemplar->input     = malloc (inodes_no * sizeof (*(exemplar->input)));
	exemplar->output    = malloc (onodes_no * sizeof (*(exemplar->output)));

	return exemplar;
}

int nn_exemplar_free (NN_exemplar *exemplar)
{
	free (exemplar->input);
	free (exemplar->output);
	free (exemplar);

	return SUCCESS;
}

NN_sample *nn_sample_new (int size)
{
	NN_sample *sample = malloc (sizeof (*sample));

	sample->size = size;
	sample->exemplar = malloc (size * sizeof (*(sample->exemplar)));

	return sample;
}

int nn_sample_renorm_from_file (NN_sample *sample, double vmin, double vmax, char *dumpfile)
{
	int i, j;
	double *pmin, *pmax;
	FILE *dump = fopen (dumpfile, "r");

	if (!dump) {
		printf ("parameter bounds file cannot be opened, aborting.\n");
		return NN_ERROR_FILENAME_INVALID;
	}

	pmin = malloc (sample->exemplar[0]->onodes_no * sizeof (*pmin));
	pmax = malloc (sample->exemplar[0]->onodes_no * sizeof (*pmax));

	for (j = 0; j < sample->exemplar[0]->onodes_no; j++)
		fscanf (dump, "%lf\t%lf\n", &(pmin[j]), &(pmax[j]));
	fclose (dump);

	for (i = 0; i < sample->size; i++)
		for (j = 0; j < sample->exemplar[i]->onodes_no; j++)
			sample->exemplar[i]->output[j] = vmin + (vmax - vmin) / (pmax[j]-pmin[j]) * (sample->exemplar[i]->output[j]-pmin[j]);

	return SUCCESS;
}

int nn_sample_renorm (NN_sample *sample, double vmin, double vmax, char *dumpfile)
{
	int i, j;

	double *pmin = malloc (sample->exemplar[0]->onodes_no * sizeof (*pmin));
	double *pmax = malloc (sample->exemplar[0]->onodes_no * sizeof (*pmax));

	if (sample->size < 2)
		return NN_ERROR_SAMPLE_INVALID_SIZE;

	for (j = 0; j < sample->exemplar[0]->onodes_no; j++)
		pmin[j] = pmax[j] = sample->exemplar[0]->output[j];

	for (i = 1; i < sample->size; i++)
		for (j = 0; j < sample->exemplar[i]->onodes_no; j++) {
			if (pmin[j] > sample->exemplar[i]->output[j]) pmin[j] = sample->exemplar[i]->output[j];
			if (pmax[j] < sample->exemplar[i]->output[j]) pmax[j] = sample->exemplar[i]->output[j];
		}

	if (VERBOSE == 1)
		for (j = 0; j < sample->exemplar[0]->onodes_no; j++)
			printf ("Output %d limits: % lf, % lf\n", j+1, pmin[j], pmax[j]);

	if (dumpfile) {
		FILE *dump = fopen (dumpfile, "w");
		if (!dump)
			printf ("file for parameter bounds cannot be opened, ignoring.\n");
		else {
			for (j = 0; j < sample->exemplar[0]->onodes_no; j++)
				fprintf (dump, "% lf\t% lf\n", pmin[j], pmax[j]);
			fclose (dump);
		}
	}

	for (i = 0; i < sample->size; i++)
		for (j = 0; j < sample->exemplar[i]->onodes_no; j++)
			sample->exemplar[i]->output[j] = vmin + (vmax - vmin) / (pmax[j]-pmin[j]) * (sample->exemplar[i]->output[j]-pmin[j]);

	return SUCCESS;
}

int nn_sample_free (NN_sample *sample)
{
	free (sample->exemplar);
	free (sample);

	return SUCCESS;
}

NN_network *nn_network_new (int layers_no, double lrp)
{
	NN_network *network = malloc (sizeof (*network));

	network->layers_no  = layers_no;
	network->lrp        = lrp;
	network->layer      = malloc (layers_no * sizeof (*(network->layer)));
	network->connection = malloc ((layers_no - 1) * sizeof (*(network->connection)));
	network->direction  = malloc ((layers_no - 1) * sizeof (*(network->direction)));

	return network;
}

int nn_network_train (NN_network *network, NN_sample *sample, double *cost_function)
{
	int i, j, k, p;
	double output_k, output_j;
	double diff_k;
	double ***delta;

	delta = malloc (sample->size * sizeof (*delta));
	for (p = 0; p < sample->size; p++) {
		delta[p] = malloc (network->layers_no * sizeof (**delta));
		for (k = 0; k < network->layers_no; k++)
			delta[p][k] = malloc (network->layer[k]->nodes_no * sizeof (***delta));
	}

	/*
	 * Initialize the cost function to be 0; this value will have only a
	 * qualitative meaning, it will not be used by the network in any way.
	 */

	*cost_function = 0.0;

	for (p = 0; p < sample->size; p++) {
		/* Fill in input values: */
		nn_layer_stimulate (network->layer[0], NN_TF_LINEAR, sample->exemplar[p]->input);

#if VERBOSE
		printf ("Pattern %d:\n  I:\t", p+1); nn_layer_print (network->layer[0]);
		printf ("  I2H:"); nn_connection_print (network->connection[0]);
#endif

		/* Propagate to the hidden layer: */
		nn_layer_propagate (network->layer[0], network->connection[0], network->direction[0], NN_TF_SIGMOID);

#if VERBOSE
		printf ("  H:\t"); nn_layer_print (network->layer[1]);
		printf ("  H2O:"); nn_connection_print (network->connection[1]);
#endif

		/* Propagate to the output layer: */
		nn_layer_propagate (network->layer[1], network->connection[1], network->direction[1], NN_TF_SIGMOID);

#if VERBOSE
		printf ("  O:\t"); nn_layer_print (network->layer[2]);
#endif

		/* Compute the output layer errors: */
		for (k = 0; k < network->layer[2]->nodes_no; k++) {
			output_k = network->layer[2]->stimulus[k];
			  diff_k = sample->exemplar[p]->output[k] - output_k;

			delta[p][2][k] = diff_k * output_k * (1.0 - output_k);
			*cost_function += diff_k * diff_k;
#if VERBOSE
			printf ("delO[%d][%d] = %lf\n", p, k, delta[p][2][k]);
			printf ("%lf -> %lf\n", sample->exemplar[p]->output[k], output_k);
#endif
		}

		for (j = 0; j < network->layer[1]->nodes_no; j++) {
			output_j = network->layer[1]->stimulus[j];

			/* Compute the hidden layer error: */
			delta[p][1][j] = 0.0;
			for (k = 0; k < network->layer[2]->nodes_no; k++)
				delta[p][1][j] += network->connection[1]->weights[k][j] * delta[p][2][k];
			delta[p][1][j] *= output_j * (1.0 - output_j);
#if VERBOSE
			printf ("delH[%d][%d] = %lf\n", p, j, delta[p][1][j]);
#endif

			/* Update the weights from hidden to output layer: */
			for (k = 0; k < network->layer[2]->nodes_no; k++)
				network->connection[1]->weights[k][j] += network->lrp * delta[p][2][k] * output_j;

			/* Update the weights from input to hidden layer: */
			for (i = 0; i < network->layer[0]->nodes_no; i++)
				network->connection[0]->weights[j][i] += network->lrp * delta[p][1][j] * network->layer[0]->stimulus[i];
		}

#if VERBOSE
	printf ("  E_p:\t%lf\n", *cost_function);
	printf ("  I2H':");
	nn_connection_print (network->connection[0]);
	printf ("  H20':");
	nn_connection_print (network->connection[1]);
#endif
	}

	/* Renormalize the cost function: */
	*cost_function /= 2.0 * network->layer[2]->nodes_no;

	for (p = 0; p < sample->size; p++) {
		for (k = 0; k < network->layers_no; k++)
			free (delta[p][k]);
		free (delta[p]);
	}
	free (delta);

	return SUCCESS;
}

int nn_network_recognize (NN_network *network, NN_sample *sample)
{
	int i, p;

	for (p = 0; p < sample->size; p++) {
		/* Fill in input values: */
		nn_layer_stimulate (network->layer[0], NN_TF_LINEAR, sample->exemplar[p]->input);

#if VERBOSE
		printf ("Pattern %d:\n  I:\t", p+1); nn_layer_print (network->layer[0]);
		printf ("  I2H:"); nn_connection_print (network->connection[0]);
#endif

		/* Propagate to the hidden layer: */
		nn_layer_propagate (network->layer[0], network->connection[0], network->direction[0], NN_TF_SIGMOID);

#if VERBOSE
		printf ("  H:\t"); nn_layer_print (network->layer[1]);
		printf ("  H2O:"); nn_connection_print (network->connection[1]);
#endif

		/* Propagate to the output layer: */
		nn_layer_propagate (network->layer[1], network->connection[1], network->direction[1], NN_TF_SIGMOID);

#if VERBOSE
		printf ("  O:\t"); nn_layer_print (network->layer[2]);
#endif

		for (i = 0; i < sample->exemplar[p]->onodes_no; i++)
			printf ("%lf\t%lf\t", network->layer[2]->stimulus[i], sample->exemplar[p]->output[i]);
		printf ("\n");
	}

	return SUCCESS;
}

int nn_network_free (NN_network *network)
{
	free (network->layer);
	free (network->connection);
	free (network->direction);
	free (network);

	return SUCCESS;
}
