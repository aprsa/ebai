#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <time.h>

#include "nn.h"

#define SUCCESS 0
#define VERBOSE 1

char *DATA_DIR;
char *DATA_FORMAT;

int MODE;

int  INPUT_UNITS;
int HIDDEN_UNITS;
int OUTPUT_UNITS;

double LRP;

int ITERS;
int ITER_OFFSET;

int SAMPLE_SIZE;

double WEIGHTS_SEED;

int ADD_JITTER;
double JITTER_SIGMA;
double JITTER_SEED;

int UNKNOWN_DATA;

int NORMALIZE_INPUT;

int DUMPSKIP;

int DISABLE_REMAPPING;
int PRINT_EXEMPLARS;

enum {
	TRAINING,
	RECOGNITION,
	CONTINUED,
	UNDEFINED
};

void print_synopsis_and_exit ()
{
	printf ("\nUsage: ./ebai -r|-t|-c [--other-switches]\n\n");
	printf ("  -h (or -?)             ..  help summary (this screen)\n");
	printf ("  -r                     ..  recognition mode\n");
	printf ("  -t                     ..  training mode\n");
	printf ("  -c                     ..  continued training mode\n");
	printf ("  -i iters               ..  number of iterations (t/c modes only)\n");
	printf ("  -s exemplars           ..  number of exemplars (t/c modes) or inputs (r mode)\n");
	printf ("  -n i:h:o               ..  network topology:\n");
	printf ("                             i is the number of input nodes\n");
	printf ("                             h is the number of hidden nodes\n");
	printf ("                             o is the number of output nodes\n");
	printf ("  --lrp value            ..  learning rate parameter value (eta)\n");
	printf ("  --data-dir dirname     ..  sample data directory (for all r/t/c modes)\n");
	printf ("  --data-format format   ..  C-style format for the filename (must have %%d)\n");
	printf ("  --unknown-data         ..  use when there is no parameter header in data\n");
	printf ("  --i2h file             ..  use 'file' for input->hidden weights\n");
	printf ("  --h2o file             ..  use 'file' for hidden->output weights\n");
	printf ("  --param-bounds file    ..  use 'file' for parameter bounds (else auto)\n");
	printf ("  --jitter sigma         ..  add synthetic jitter to sample data\n");
	printf ("  --jitter-seed seed     ..  seed to use for randomizing jitter\n");
	printf ("  --weights-seed seed    ..  seed to use for randomizing weights\n");
	printf ("  --iter-offset N        ..  iteration offset for continued training\n");
	printf ("  --dump-i2h prefix      ..  dump I2H weights for each iteration\n");
	printf ("  --dump-h2o prefix      ..  dump H2O weights for each iteration\n");
	printf ("  --dump-skip N          ..  dump weights every N-th iteration\n");
	printf ("  --normalize-input      ..  map the input to the [0,1] interval\n");
	printf ("  --disable-remapping    ..  disable remapping of outputs to [0.1,0.9]\n");
	printf ("  --print-exemplars      ..  print exemplars on screen (i.e. debug)\n");
	printf ("\n");
	exit (0);
}

int read_in_parameters (char *filename, NN_exemplar *exemplar)
{
	int i;
	FILE *in = fopen (filename, "r");
	if (!in) {
		printf ("*** error: data file %s cannot be opened!\n", filename);
		exit (0);
	}

	/* Read input unit values: */
	for (i = 0; i < INPUT_UNITS; i++)
		fscanf (in, "%lf\n", &(exemplar->input[i]));

	/* Read output unit values: */
	if (!UNKNOWN_DATA)
		for (i = 0; i < OUTPUT_UNITS; i++)
			fscanf (in, "%lf\n", &(exemplar->output[i]));
	else
		for (i = 0; i < OUTPUT_UNITS; i++)
			exemplar->output[i] = 0.5;

	fclose (in);

	return SUCCESS;
}

int normalize_input (NN_exemplar *exemplar)
{
	/*
	 * This function will map the light curve to the [0, 1] interval.
	 */

	int i;
	double pmin, pmax;

	pmin = pmax = exemplar->input[0];
	for (i = 1; i < exemplar->inodes_no; i++) {
		if (pmin > exemplar->input[i]) pmin = exemplar->input[i];
		if (pmax < exemplar->input[i]) pmax = exemplar->input[i];
	}

	for (i = 0; i < exemplar->inodes_no; i++)
		exemplar->input[i] = (exemplar->input[i] - pmin) / (pmax - pmin);

	return SUCCESS;
}

int add_noise (NN_exemplar *exemplar, double sigma)
{
	/*
	 * This function adds synthetic Gaussian noise to the exemplar.
	 */

	int i;
	double u, r;

	for (i = 0; i < exemplar->inodes_no; i++) {
		u = (double) rand () / RAND_MAX;
		r = sigma * sqrt ( 2.0 * log (1.0/(1.0-u)) );
		u = (double) rand () / RAND_MAX;
		exemplar->input[i] += r * cos (2*M_PI*u);
	}

	return SUCCESS;
}

int read_in_weights (char *filename, NN_connection *connection)
{
	int i, j;
	FILE *in = fopen (filename, "r");

	for (i = 0; i < connection->to->nodes_no; i++) {
		for (j = 0; j < connection->from->nodes_no; j++)
			fscanf (in, "%lf\t", &(connection->weights[i][j]));
		fscanf (in, "\n");
	}

	fclose (in);

	return SUCCESS;
}

int main (int argc, char **argv)
{
	int i;

	NN_layer *input, *hidden, *output;
	NN_connection *input2hidden, *hidden2output;
	NN_network *network;

	NN_sample *sample;
	double cf; /* Cost function */

	char readout_str[255];
	char i2h_filename[255], h2o_filename[255];
	char file_format[255];

	char *param_output_file   = NULL;
	char *i2h_dump_prefix     = NULL;
	char *h2o_dump_prefix     = NULL;

	double *data = malloc (INPUT_UNITS * sizeof (*data));

	char filename[255];

	if (argc < 2)
		print_synopsis_and_exit ();

	/* Set the defaults: */
	DATA_DIR = strdup ("data");
	DATA_FORMAT = strdup ("dlc%d.data");

	MODE = UNDEFINED;

	INPUT_UNITS  = 201;
	HIDDEN_UNITS =  20;
	OUTPUT_UNITS =   5;

	LRP = 0.15;

	ITER_OFFSET = 0;
	ITERS = 99999;

	SAMPLE_SIZE = 2000;

	ADD_JITTER = 0;
	JITTER_SIGMA = 0.0;
	JITTER_SEED = time (0);

	UNKNOWN_DATA = 0;

	WEIGHTS_SEED = time (0);

	NORMALIZE_INPUT = 0;

	DUMPSKIP = 1;

	DISABLE_REMAPPING = 0;
	PRINT_EXEMPLARS = 0;

	sprintf (i2h_filename, "i2h.weights");
	sprintf (h2o_filename, "h2o.weights");

	/* Parse the command line: */
	for (i = 1; i < argc; i++) {
		if (strcmp (argv[i], "-h") == 0 || strcmp (argv[i], "-?") == 0)
			print_synopsis_and_exit ();
		else if (strcmp (argv[i], "-r") == 0)
			MODE = RECOGNITION;
		else if (strcmp (argv[i], "-t") == 0)
			MODE = TRAINING;
		else if (strcmp (argv[i], "-c") == 0)
			MODE = CONTINUED;
		else if (strcmp (argv[i], "-i") == 0) {
			i++;
			sscanf (argv[i], "%d", &ITERS);
		}
		else if (strcmp (argv[i], "-s") == 0) {
			i++;
			sscanf (argv[i], "%d", &SAMPLE_SIZE);
		}
		else if (strcmp (argv[i], "-n") == 0) {
			i++;
			sscanf (argv[i], "%d:%d:%d", &INPUT_UNITS, &HIDDEN_UNITS, &OUTPUT_UNITS);
		}
		else if (strcmp (argv[i], "--lrp") == 0) {
			i++;
			sscanf (argv[i], "%lf", &LRP);
		}
		else if (strcmp (argv[i], "--data-dir") == 0) {
			i++;
			sscanf (argv[i], "%s", &(readout_str[0]));
			free (DATA_DIR);
			DATA_DIR = strdup (readout_str);
		}
		else if (strcmp (argv[i], "--data-format") == 0) {
			i++;
			sscanf (argv[i], "%s", &(readout_str[0]));
			free (DATA_FORMAT);
			DATA_FORMAT = strdup (readout_str);
		}
		else if (strcmp (argv[i], "--unknown-data") == 0) {
			UNKNOWN_DATA = 1;
		}
		else if (strcmp (argv[i], "--i2h") == 0) {
			i++;
			sscanf (argv[i], "%s", &(i2h_filename[0]));
		}
		else if (strcmp (argv[i], "--h2o") == 0) {
			i++;
			sscanf (argv[i], "%s", &(h2o_filename[0]));
		}
		else if (strcmp (argv[i], "--param-bounds") == 0) {
			i++;
			sscanf (argv[i], "%s", &(readout_str[0]));
			param_output_file = strdup (readout_str);
		}
		else if (strcmp (argv[i], "--jitter") == 0) {
			ADD_JITTER = 1;
			i++;
			sscanf (argv[i], "%lf", &JITTER_SIGMA);
		}
		else if (strcmp (argv[i], "--jitter-seed") == 0) {
			i++;
			sscanf (argv[i], "%lf", &JITTER_SEED);
		}
		else if (strcmp (argv[i], "--weights-seed") == 0) {
			i++;
			sscanf (argv[i], "%lf", &WEIGHTS_SEED);
		}
		else if (strcmp (argv[i], "--dump-i2h") == 0) {
			i++;
			sscanf (argv[i], "%s", &(readout_str[0]));
			i2h_dump_prefix = strdup (readout_str);
		}
		else if (strcmp (argv[i], "--dump-h2o") == 0) {
			i++;
			sscanf (argv[i], "%s", &(readout_str[0]));
			h2o_dump_prefix = strdup (readout_str);
		}
		else if (strcmp (argv[i], "--dump-skip") == 0) {
			i++;
			sscanf (argv[i], "%d", &DUMPSKIP);
		}
		else if (strcmp (argv[i], "--iter-offset") == 0) {
			i++;
			sscanf (argv[i], "%d", &ITER_OFFSET);
		}
		else if (strcmp (argv[i], "--normalize-input") == 0) {
			NORMALIZE_INPUT = 1;
		}
		else if (strcmp (argv[i], "--disable-remapping") == 0) {
			DISABLE_REMAPPING = 1;
		}
		else if (strcmp (argv[i], "--print-exemplars") == 0) {
			PRINT_EXEMPLARS = 1;
		}
		else {
			printf ("\nSwitch %s not recognized, aborting.\n\n", argv[i]);
			exit (0);
		}
	}

#if VERBOSE
	printf ("# Issued command:\n");
	printf ("#   ");
	for (i = 0; i < argc; i++)
		printf ("%s ", argv[i]);
	printf ("\n");

	printf ("# building a neural network: %d input nodes, %d hidden nodes, %d output nodes\n", INPUT_UNITS, HIDDEN_UNITS, OUTPUT_UNITS);
	printf ("# learning rate parameter (eta): %lf\n", LRP);
#endif

	/* Create the network topology: */
	input  = nn_layer_new (INPUT_UNITS);
	hidden = nn_layer_new (HIDDEN_UNITS);
	output = nn_layer_new (OUTPUT_UNITS);

	/* Create the network connections: */
	input2hidden  = nn_connection_new (input, hidden);
	hidden2output = nn_connection_new (hidden, output);

	if (MODE == RECOGNITION) {
		printf ("# starting the program in recognition mode\n");
		printf ("# reading in weights:\n");
		printf ("#   input->hidden weights read from %s\n", i2h_filename);
		printf ("#   hidden->output weights read from %s\n", h2o_filename);
		read_in_weights (i2h_filename, input2hidden);
		read_in_weights (h2o_filename, hidden2output);
	}
	if (MODE == CONTINUED) {
		printf ("# starting the program in continued training mode\n");
		printf ("# reading in weights:\n");
		printf ("#   input->hidden weights read from %s\n", i2h_filename);
		printf ("#   hidden->output weights read from %s\n", h2o_filename);
		read_in_weights (i2h_filename, input2hidden);
		read_in_weights (h2o_filename, hidden2output);
	}
	if (MODE == TRAINING) {
		printf ("# starting the program in training mode\n");
		printf ("# randomizing weights ... ");
		srand (WEIGHTS_SEED);
		nn_connection_randomize (input2hidden);
		nn_connection_randomize (hidden2output);
		printf ("done\n");
	}

	/* Assemble the network: */
	network                = nn_network_new (3, LRP);
	network->layer[0]      = input;
	network->layer[1]      = hidden;
	network->layer[2]      = output;
	network->connection[0] = input2hidden;
	network->connection[1] = hidden2output;
	network->direction[0]  = NN_FORWARD;
	network->direction[1]  = NN_FORWARD;

	if (MODE == RECOGNITION) {
		if (UNKNOWN_DATA)
			printf ("# reading in %d unknown datasets\n", SAMPLE_SIZE);
		else
			printf ("# reading in %d test datasets\n", SAMPLE_SIZE);

		if (NORMALIZE_INPUT)
			printf ("# mapping the data to the [0,1] interval\n");
		if (ADD_JITTER)
			printf ("# adding %g%% synthetic noise to the data\n", 100.0 * JITTER_SIGMA);

		sample = nn_sample_new (SAMPLE_SIZE);

		sprintf (file_format, "%s/%s", DATA_DIR, DATA_FORMAT);
		for (i = 0; i < SAMPLE_SIZE; i++) {
			sprintf (filename, file_format, i+1);
			sample->exemplar[i] = nn_exemplar_new (INPUT_UNITS, OUTPUT_UNITS);
			read_in_parameters (filename, sample->exemplar[i]);
			if (PRINT_EXEMPLARS) {
				int _i;
				for (_i = 0; _i < INPUT_UNITS; _i++)
					printf ("% 10.6lf", sample->exemplar[i]->input[_i]);
				printf ("\n");
				for (_i = 0; _i < OUTPUT_UNITS; _i++)
					printf ("% 10.6lf", sample->exemplar[i]->output[_i]);
				printf ("\n");
			}
			if (NORMALIZE_INPUT)
				normalize_input (sample->exemplar[i]);
			if (ADD_JITTER) {
				srand (JITTER_SEED);
				add_noise (sample->exemplar[i], JITTER_SIGMA);
			}
		}

		if (!UNKNOWN_DATA) {
			if (!DISABLE_REMAPPING) {
#if VERBOSE
				printf ("# remapping exemplars to [0.1,0.9] interval\n");
				if (param_output_file)
					printf ("#   reading the parameters from %s\n", param_output_file);
#endif
				if (param_output_file)
					nn_sample_renorm_from_file (sample, 0.1, 0.9, param_output_file);
				else
					nn_sample_renorm (sample, 0.1, 0.9, param_output_file);
			}
		}

		printf ("# initiating recognition\n");
		nn_network_recognize (network, sample);
	}

	if (MODE == TRAINING || MODE == CONTINUED) {
#if VERBOSE
		printf ("# reading in %d exemplars\n", SAMPLE_SIZE);
#endif

		if (NORMALIZE_INPUT)
			printf ("# mapping sample data to the [0,1] interval\n");
		if (ADD_JITTER)
			printf ("# adding %g%% synthetic noise to the data\n", 100.0 * JITTER_SIGMA);

		sample = nn_sample_new (SAMPLE_SIZE);

		sprintf (file_format, "%s/%s", DATA_DIR, DATA_FORMAT);
		for (i = 0; i < SAMPLE_SIZE; i++) {
			sprintf (filename, file_format, i+1);
			sample->exemplar[i] = nn_exemplar_new (INPUT_UNITS, OUTPUT_UNITS);
			read_in_parameters (filename, sample->exemplar[i]);
			if (PRINT_EXEMPLARS) {
				int _i;
				for (_i = 0; _i < INPUT_UNITS; _i++)
					printf ("% 10.6lf", sample->exemplar[i]->input[_i]);
				printf ("\n");
				for (_i = 0; _i < OUTPUT_UNITS; _i++)
					printf ("% 10.6lf", sample->exemplar[i]->output[_i]);
				printf ("\n");
			}
			if (NORMALIZE_INPUT)
				normalize_input (sample->exemplar[i]);
			if (ADD_JITTER) {
				srand (JITTER_SEED);
				add_noise (sample->exemplar[i], JITTER_SIGMA);
			}
		}

		if (!DISABLE_REMAPPING) {
#if VERBOSE
			printf ("# renormalizing network parameters\n");
			if (param_output_file)
				printf ("#   writing the parameters to %s\n", param_output_file);
#endif
			nn_sample_renorm (sample, 0.1, 0.9, param_output_file);
		}

#if VERBOSE
		if (MODE == TRAINING)
			printf ("# training the network for %d iterations:\n", ITERS);
		else
			printf ("# continued training for another %d iterations:\n", ITERS);

		printf ("# iter\t\tcf\n");
#endif
		for (i = 1; i <= ITERS; i++) {
			nn_network_train (network, sample, &cf);
			printf ("%5d\t%e\n", ITER_OFFSET + i, cf);
			if (i2h_dump_prefix && i % DUMPSKIP == 0) {
				sprintf (filename, "%s%04d.weights", i2h_dump_prefix, ITER_OFFSET + i);
				nn_connection_print_to_file (filename, input2hidden);
			}
			if (h2o_dump_prefix && i % DUMPSKIP == 0) {
				sprintf (filename, "%s%04d.weights", h2o_dump_prefix, ITER_OFFSET + i);
				nn_connection_print_to_file (filename, hidden2output);
			}
		}

#if VERBOSE
		printf ("# writing weights to disk\n");
#endif
		nn_connection_print_to_file (i2h_filename, input2hidden);
		nn_connection_print_to_file (h2o_filename, hidden2output);
	}

#if VERBOSE
	printf ("# freeing the network and exitting\n");
#endif

	nn_network_free (network);

	for (i = 0; i < SAMPLE_SIZE; i++)
		nn_exemplar_free (sample->exemplar[i]);
	nn_sample_free (sample);

	nn_connection_free (input2hidden);
	nn_connection_free (hidden2output);

	nn_layer_free (input);
	nn_layer_free (hidden);
	nn_layer_free (output);

	free (DATA_DIR);
	free (data);

	return SUCCESS;
}
